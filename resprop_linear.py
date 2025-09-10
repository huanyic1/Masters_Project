import torch
import torch.nn as nn
import copy
import math


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
        pad = (group_size - grad_diff.shape[-1] % group_size) % group_size
        if pad:
            grad_diff = torch.nn.functional.pad(grad_diff, (0, pad))

        G = grad_diff.shape[-1] // group_size
        x = grad_diff.view(*grad_diff.shape[:-1], G, group_size)
        ax = x.abs()

        # nth largest == kth smallest of (-abs) with k=n
        kth_neg, _ = (-ax).kthvalue(k=n, dim=-1, keepdim=True)   # (..., G, 1)
        thresh = -kth_neg                                         # (..., G, 1)

        # mask in-place without allocating zeros_like
        out = x.clone()
        out.masked_fill_(ax < thresh, 0)

        out = out.view(*grad_diff.shape)
        if pad:
            out = out[..., :-pad]
        return out
    else:
        N = grad_diff.numel()
        k_keep = int(max(0, math.floor(reuse_percentage * N)))
        if k_keep <= 0:
            return torch.zeros_like(grad_diff)
        if k_keep >= N:
            return grad_diff

        abs_flat = grad_diff.abs().view(-1)
        vals, idx = torch.topk(abs_flat, k_keep, largest=True, sorted=False)
        out = torch.zeros_like(grad_diff).view(-1)
        out.scatter_(0, idx, grad_diff.view(-1).index_select(0, idx))
        return out.view_as(grad_diff)

class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, reuse_percentage, structured=False, n=2, group_size=4):
        if prev_grad_output is not None and reuse_percentage > 0 and \
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
            reuse_schedule: list = [(0.9, 0)], 
            avg: bool = True
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.prev_gradients = None
        self.reuse_schedule = reuse_schedule
        self.step_counter = {}
        self.avg = avg

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
                if reuse_percentage > 0 and self.step_counter[device] % self.k == 0:
                    if self.avg:
                        self.prev_gradients[device] = grad_output.sum(dim=0) / grad_output.size(0) #torch.mean(grad_output, dim=0) # 
                    else: 
                        self.prev_gradients[device] = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
            output.register_hook(hook)
        else:
            self.prev_gradients = None

        return output

    def extra_repr(self):
        return super().extra_repr() + f", reuse_percentage={self.reuse_percentage}"


def respropify_bert(base_model, reuse_schedule=[(0.9, 0)]):
    def resprop_linear(layer: nn.Linear):
        new_layer = ReSpropLinear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            reuse_schedule=reuse_schedule
        )
        new_layer.weight.data.copy_(layer.weight.data)
        if layer.weight.grad is not None:
            new_layer.weight.grad.data.copy_(layer.weight.grad.data)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
            if layer.bias.grad is not None:
                new_layer.bias.grad.data.copy_(layer.bias.grad.data)
        return new_layer

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