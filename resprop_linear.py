import torch
import torch.nn as nn
import copy
import torch.distributed as dist

class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, reuse_percentage):
        device = input.device
        weight, bias = weight.to(device), bias.to(device) if bias is not None else None
        prev_grad_output = prev_grad_output.to(device) if prev_grad_output is not None else None

        if prev_grad_output is not None and len(input.shape) == 3 and prev_grad_output.size(0) == input.size(1):
            prev_grad_input = torch.mm(prev_grad_output, weight)
            prev_grad_weight = torch.mm(prev_grad_output.t(), torch.sum(input, dim=0))
        else:
            prev_grad_output = prev_grad_input = prev_grad_weight = None

        ctx.reuse_percentage = reuse_percentage
        ctx.save_for_backward(input, weight, bias, prev_grad_output, prev_grad_input, prev_grad_weight)

        output = torch.bmm(input, weight.t().expand(input.size(0), -1, -1))
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, prev_grad_output, prev_grad_input, prev_grad_weight = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        # Compute reuse mask
        if prev_grad_output is not None:
            grad_diff = grad_output - prev_grad_output

            if ctx.reuse_percentage > 0:
                threshold_idx = int(len(grad_diff.flatten()) * ctx.reuse_percentage)
                threshold = torch.kthvalue(torch.abs(grad_diff).flatten(), threshold_idx)[0]

                # Sparsify grad_output
                reuse_mask = torch.abs(grad_diff) <= threshold
                grad_diff = torch.where(reuse_mask, torch.tensor(0, device=grad_diff.device), grad_diff)

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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, reuse_percentage: float = 0.9):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.reuse_percentage = reuse_percentage
        self.prev_gradients = {}

    def forward(self, input):
        device = input.device
        if device not in self.prev_gradients:
            self.prev_gradients[device] = None  # Store per-device gradients

        output = ReSpropLinearFunction.apply(
            input, self.weight.to(device),
            self.bias.to(device) if self.bias is not None else None,
            self.prev_gradients[device], self.reuse_percentage
        )

        if output.requires_grad:
            def hook(grad_output):
                self.prev_gradients[device] = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
                return None

            output.register_hook(hook)
        else:
            self.prev_gradients[device] = None

        return output

    def extra_repr(self):
        return super().extra_repr() + f", reuse_percentage={self.reuse_percentage}"


class ReSpropWarmupLinear(ReSpropLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, reuse_schedule=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        if reuse_schedule is None:
            raise ValueError("reuse_schedule must be provided as a list of (iteration, reuse_percentage) tuples")

        # Validate and sort the schedule
        self.reuse_schedule = sorted(reuse_schedule, key=lambda x: x[0])
        self.iteration = 0

    def get_current_reuse_percentage(self):
        for i in range(len(self.reuse_schedule) - 1, -1, -1):
            if self.iteration >= self.reuse_schedule[i][0]:
                return self.reuse_schedule[i][1]
        return 0.0  # Default if no match

    def forward(self, input):
        device = input.device
        if device not in self.prev_gradients:
            self.prev_gradients[device] = None

        reuse_percentage = self.get_current_reuse_percentage()

        output = ReSpropLinearFunction.apply(
            input, self.weight.to(device),
            self.bias.to(device) if self.bias is not None else None,
            self.prev_gradients[device], reuse_percentage
        )

        if output.requires_grad:
            def hook(grad_output):
                self.prev_gradients[device] = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
                return None

            output.register_hook(hook)
        else:
            self.prev_gradients[device] = None

        self.iteration += 1
        return output

    def extra_repr(self):
        return super(ReSpropLinear, self).extra_repr() + f", reuse_schedule={self.reuse_schedule}"


def resprofify_bert(base_model, reuse_percentage=0.9):
    model = copy.deepcopy(base_model).to(torch.cuda.current_device())

    def resprop_linear(layer: nn.Linear):
        return ReSpropLinear(layer.in_features, layer.out_features, layer.bias is not None, reuse_percentage=reuse_percentage)

    for layer in model.bert.encoder.layer:
        att = layer.attention
        att.self.query = resprop_linear(att.self.query)
        att.self.key = resprop_linear(att.self.key)
        att.self.value = resprop_linear(att.self.value)
        att.output.dense = resprop_linear(att.output.dense)

        # Feed Forward Block
        layer.intermediate.dense = resprop_linear(layer.intermediate.dense)
        layer.output.dense = resprop_linear(layer.output.dense)

    return model



def resprofify_bert_warmup(base_model, reuse_schedule=[(0,0.9)]):
    model = copy.deepcopy(base_model).to(torch.cuda.current_device())

    def resprop_linear(layer: nn.Linear):
        return ReSpropWarmupLinear(layer.in_features, layer.out_features, layer.bias is not None, reuse_schedule=reuse_schedule)

    for layer in model.bert.encoder.layer:
        att = layer.attention
        att.self.query = resprop_linear(att.self.query)
        att.self.key = resprop_linear(att.self.key)
        att.self.value = resprop_linear(att.self.value)
        att.output.dense = resprop_linear(att.output.dense)

        # Feed Forward Block
        layer.intermediate.dense = resprop_linear(layer.intermediate.dense)
        layer.output.dense = resprop_linear(layer.output.dense)

    return model