import torch
import torch.nn as nn
import copy


def get_current_reuse_percentage(reuse_schedule, step): 
    """
    Assumes that for unstructured, all reuse percentages passed in as a number
    Assumes that for structured, all reuse percentages passed in as a string
    """
    applicable = [rp for rp, start in reuse_schedule if step >= start]
    reuse_percentage = applicable[-1] if applicable else 0.0
    if isinstance(reuse_percentage, (int, float)):
        return reuse_percentage, None, False
    elif isinstance(reuse_percentage, str):
        parts = reuse_percentage.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid ratio string format: '{reuse_percentage}' (expected 'num / den')")
        num = int(parts[0].strip())
        den = int(parts[1].strip())
        return num, den, True
    raise ValueError(f"Invalid reuse percentage format: '{reuse_percentage}'")


def generate_reuse_mask(reuse_percentage, grad_output, prev_grad_output, structured=False, n=2, group_size=4):
    grad_diff = grad_output - prev_grad_output

    if reuse_percentage == 0:
        return grad_diff

    if structured:
        shape = grad_diff.shape
        last_dim = shape[-1]
        remainder = last_dim % group_size
        pad_needed = (group_size - remainder) if remainder != 0 else 0

        # Pad the last dimension if necessary
        if pad_needed > 0:
            pad_shape = list(shape[:-1]) + [pad_needed]
            pad_tensor = torch.zeros(pad_shape, device=grad_diff.device, dtype=grad_diff.dtype)
            grad_diff = torch.cat([grad_diff, pad_tensor], dim=-1)

        # Reshape into (..., num_groups, group_size)
        new_last_dim = grad_diff.shape[-1]
        num_groups = new_last_dim // group_size
        new_shape = grad_diff.shape[:-1] + (num_groups, group_size)
        grad_diff_grouped = grad_diff.view(*new_shape)

        # Compute top-k mask within each group
        abs_vals = grad_diff_grouped.abs()
        topk = torch.topk(abs_vals, k=n, dim=-1)
        threshold = topk.values.min(dim=-1, keepdim=True).values  # shape (..., G, 1)
        reuse_mask = abs_vals >= threshold

        # Apply the mask
        masked_grad = torch.where(reuse_mask, grad_diff_grouped, torch.zeros_like(grad_diff_grouped))

        # Reshape back to original padded shape, then trim off padding
        masked_grad = masked_grad.view(*grad_diff.shape)
        if pad_needed > 0:
            masked_grad = masked_grad[..., :-pad_needed]

        return masked_grad

    else:
        # Unstructured sparsity (elementwise top-k)
        abs_grad_diff = torch.abs(grad_diff)
        flat = abs_grad_diff.flatten()
        threshold_idx = int(flat.size(0) * (1 - reuse_percentage))
        threshold = torch.topk(flat, threshold_idx, largest=True).values[-1]
        mask = abs_grad_diff > threshold
        return torch.where(mask, grad_diff, torch.tensor(0., device=grad_diff.device))

class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, reuse_percentage, structured=False, n=2, group_size=4):
        if prev_grad_output is not None and \
                len(input.shape) == 3 and prev_grad_output.size(0) == input.size(1):
            prev_grad_input = torch.mm(prev_grad_output, weight)                        # pre∇w_l
            prev_grad_weight = torch.mm(prev_grad_output.t(), torch.sum(input, dim=0))  # pre∇a_l
        else:
            if prev_grad_output is not None:
                print("Warning: Couldn't reuse gradient due to shape mis-match.")

            prev_grad_output = None
            prev_grad_input = None
            prev_grad_weight = None

        ctx.reuse_percentage = reuse_percentage
        ctx.structured = structured
        ctx.n = n
        ctx.group_size = group_size

        ctx.save_for_backward(
            input, weight, bias,
            prev_grad_output, prev_grad_input, prev_grad_weight
        )

        output = torch.bmm(input, weight.t().expand(input.size(0), -1, -1))
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, prev_grad_output, prev_grad_input, prev_grad_weight = ctx.saved_tensors

        grad_input = None
        grad_weight = None
        grad_bias = None

        if prev_grad_output is not None:
            grad_diff = generate_reuse_mask(ctx.reuse_percentage, grad_output, prev_grad_output, ctx.structured, ctx.n, ctx.group_size)
            grad_output = grad_diff

        # Compute gradients
        if ctx.needs_input_grad[0]:
            grad_input = torch.bmm(grad_output, weight.expand(grad_output.size(0), -1, -1))
            if prev_grad_output is not None:
                grad_input += prev_grad_input

        if ctx.needs_input_grad[1]:
            grad_weight = torch.sum(torch.bmm(grad_output.transpose(1, 2), input), dim=0)
            if prev_grad_output is not None:
                grad_weight += prev_grad_weight

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 1))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class ReSpropLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
            reuse_schedule: list = [(0.9, 0)]
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.prev_gradients = None
        self.reuse_schedule = reuse_schedule
        self.step_counter = {}

    def forward(self, input):
        device = input.device
        self.step_counter.setdefault(device, 0)
        step = self.step_counter[device]
        num1, num2, structured = get_current_reuse_percentage(self.reuse_schedule, step)
        if not structured:
            reuse_percentage = num1
            n = None
            group_size = None
        else:
            reuse_percentage = None
            n = num1
            group_size = num2

        output = ReSpropLinearFunction.apply(input, self.weight, self.bias, self.prev_gradients, reuse_percentage, structured, n, group_size)

        if output.requires_grad:
            def hook(grad_output):
                # Store gradients for next iteration
                self.prev_gradients = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
                return None

            output.register_hook(hook)
        else:
            self.prev_gradients = None

        return output

    def extra_repr(self):
        return super().extra_repr() + f", reuse_percentage={self.reuse_percentage}"


def respropify_bert(base_model, reuse_schedule=[(0.9, 0)]):
    def resprop_linear(layer: nn.Linear):
        return ReSpropLinear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            reuse_schedule=reuse_schedule
        )

    model = copy.deepcopy(base_model).to(torch.cuda.current_device())
    for layer in model.bert.encoder.layer:
        # Self Attention
        att = layer.attention
        att.self.query = resprop_linear(att.self.query)
        att.self.key = resprop_linear(att.self.key)
        att.self.value = resprop_linear(att.self.value)
        att.output.dense = resprop_linear(att.output.dense)

        # Feed Forward Block
        layer.intermediate.dense = resprop_linear(layer.intermediate.dense)
        layer.output.dense = resprop_linear(layer.output.dense)

    return model