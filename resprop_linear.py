import torch
import torch.nn as nn
import copy


class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, reuse_percentage):
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

        # Compute reuse mask
        if prev_grad_output is not None:
            grad_diff = grad_output - prev_grad_output

            if ctx.reuse_percentage > 0:
                # Find threshold
                if grad_diff.device.type == "mps":
                    sorted_diffs = torch.sort(torch.abs(grad_diff).flatten())[0]
                    threshold_idx = int(len(sorted_diffs) * ctx.reuse_percentage)
                    threshold = sorted_diffs[threshold_idx]
                else:
                    threshold_idx = int(len(grad_diff.flatten()) * ctx.reuse_percentage)
                    threshold = torch.kthvalue(torch.abs(grad_diff).flatten(), threshold_idx)[0]

                # Sparsify grad_output
                reuse_mask = torch.abs(grad_diff) <= threshold
                grad_diff = torch.where(reuse_mask, torch.tensor(0), grad_diff)

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

        return grad_input, grad_weight, grad_bias, None, None


class ReSpropLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
            reuse_percentage: float = 0.9
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.prev_gradients = None
        self.reuse_percentage = reuse_percentage

    def forward(self, input):
        output = ReSpropLinearFunction.apply(input, self.weight, self.bias, self.prev_gradients, self.reuse_percentage)

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



def resprop_linear(layer: nn.Linear):
    return ReSpropLinear(
        layer.in_features,
        layer.out_features,
        layer.bias is not None,
        reuse_percentage=reuse_percentage
    )

def resprofify_gpt2(base_model, reuse_percentage=0.9):
    model = copy.deepcopy(base_model)
    for block in model.transformer.h:  # Iterate through transformer layers
        # Self Attention
        attn = block.attn
        attn.c_attn = resprop_linear(attn.c_attn)  # Projects query, key, and value
        attn.c_proj = resprop_linear(attn.c_proj)  # Attention output projection

        # Feed Forward Block
        block.mlp.c_fc = resprop_linear(block.mlp.c_fc)  # First Linear Layer in MLP
        block.mlp.c_proj = resprop_linear(block.mlp.c_proj)  # Second Linear Layer in MLP

    return model

def resprofify_bert(base_model, reuse_percentage=0.9):
    def resprop_linear(layer: nn.Linear):
        return ReSpropLinear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            reuse_percentage=reuse_percentage
        )
    model = copy.deepcopy(base_model)
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