import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import types

def get_current_reuse_percentage(reuse_schedule, step):
    applicable = [rp for rp, start in reuse_schedule if step >= start]
    return applicable[-1] if applicable else 0.0

class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, prev_grad_input, prev_grad_weight, reuse_percentage):
        prev_grad_output = prev_grad_output if prev_grad_output is not None else None
        prev_grad_input = prev_grad_input if prev_grad_input is not None else None
        prev_grad_weight = prev_grad_weight if prev_grad_weight is not None else None

        if prev_grad_output is not None and len(input.shape) == 3 and prev_grad_output.size(0) == input.size(1):
            pass
        else:
            if prev_grad_output is not None:
                print("Warning: Couldn't reuse gradient due to shape mis-match.")
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
                abs_grad_diff = torch.abs(grad_diff)
                threshold = torch.quantile(abs_grad_diff, ctx.reuse_percentage)
                # Sparsify grad_output
                reuse_mask = abs_grad_diff <= threshold
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

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class ReSpropLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, reuse_schedule=None, k=1):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.reuse_schedule = reuse_schedule or [(0.9, 0)]
        self.k = k
        self.prev_gradients = {}
        self.prev_grad_input = {}
        self.prev_grad_weight = {}
        self.step_counter = {}

    def forward(self, input):
        device = input.device
        self.prev_gradients.setdefault(device, None)
        self.prev_grad_input.setdefault(device, None)
        self.prev_grad_weight.setdefault(device, None)
        self.step_counter.setdefault(device, 0)

        step = self.step_counter[device]
        reuse_percentage = get_current_reuse_percentage(self.reuse_schedule, step)

        # prev_reuse_percentage = get_current_reuse_percentage(self.reuse_schedule, step - 1)
        # if reuse_percentage != prev_reuse_percentage and (device.index is None or device.index == 0):
        #     print('Switching REUSE_PERCENTAGE from', prev_reuse_percentage, 'to', reuse_percentage)

        output = ReSpropLinearFunction.apply(
            input, self.weight,
            self.bias if self.bias is not None else None,
            self.prev_gradients[device],
            self.prev_grad_input[device],
            self.prev_grad_weight[device],
            reuse_percentage, 
        )

        if output.requires_grad:
            def hook(grad_output):
                if self.step_counter[device] % self.k == 0:
                    prev_grad_output = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
                    prev_grad_input = torch.mm(prev_grad_output, self.weight)
                    prev_grad_weight = torch.mm(prev_grad_output.t(), torch.sum(input, dim=0))
                    sampled = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
                    self.prev_gradients[device] = prev_grad_output
                    self.prev_grad_input[device] = prev_grad_input
                    self.prev_grad_weight[device] = prev_grad_weight     
                self.step_counter[device] += 1
                
            output.register_hook(hook)

        return output

    def extra_repr(self):
        return super().extra_repr() + f", reuse_percentage={self.reuse_percentage}"

        
class ReSpropAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, prev_grad_Q, prev_grad_K, prev_grad_V, reuse_percentage):
        ctx.reuse_percentage = reuse_percentage
        ctx.save_for_backward(Q, K, V, prev_grad_Q, prev_grad_K, prev_grad_V)
        d_k = Q.size(-1)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        ctx.attn_probs = attn_probs

        return torch.bmm(attn_probs, V)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, prev_grad_Q, prev_grad_K, prev_grad_V = ctx.saved_tensors
        attn_probs = ctx.attn_probs
        d_k = Q.size(-1)

        grad_attn_probs = torch.bmm(grad_output, V.transpose(1, 2))
        grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_output)

        grad_attn_scores = grad_attn_probs * attn_probs
        grad_attn_scores -= attn_probs * grad_attn_scores.sum(dim=-1, keepdim=True)
        grad_attn_scores /= (d_k ** 0.5)

        grad_Q = torch.bmm(grad_attn_scores, K)
        grad_K = torch.bmm(grad_attn_scores.transpose(1, 2), Q)

        # Apply reuse
        def sparsify(current, previous):
            grad_diff = current - previous
            if ctx.reuse_percentage > 0:
                threshold_idx = int(len(grad_diff.flatten()) * ctx.reuse_percentage)
                threshold = torch.quantile(torch.abs(grad_diff), ctx.reuse_percentage)
                reuse_mask = torch.abs(grad_diff) <= threshold
                grad_diff = torch.where(reuse_mask, torch.tensor(0, device=grad_diff.device), grad_diff)
            return grad_diff + previous

        if prev_grad_Q is not None:
            grad_Q = sparsify(grad_Q, prev_grad_Q)
        if prev_grad_K is not None:
            grad_K = sparsify(grad_K, prev_grad_K)
        if prev_grad_V is not None:
            grad_V = sparsify(grad_V, prev_grad_V)

        return grad_Q, grad_K, grad_V, None, None, None, None
   

def resprop_linear(layer: nn.Linear, reuse_schedule=None, k=1):
    return ReSpropLinear(layer.in_features, layer.out_features, layer.bias is not None, reuse_schedule=reuse_schedule, k=k)
class ReSpropAttention(nn.Module):
    def __init__(self, embed_dim, reuse_schedule=None, lin_reuse_schedule=None, att_k=1, lin_k=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.reuse_schedule = reuse_schedule or [(0.9, 0)]
        self.k = att_k

        if lin_reuse_schedule:
            self.q_proj = resprop_linear(nn.Linear(embed_dim, embed_dim), reuse_schedule=lin_reuse_schedule, k=lin_k)
            self.k_proj = resprop_linear(nn.Linear(embed_dim, embed_dim), reuse_schedule=lin_reuse_schedule, k=lin_k)
            self.v_proj = resprop_linear(nn.Linear(embed_dim, embed_dim), reuse_schedule=lin_reuse_schedule, k=lin_k)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.prev_gradients = {}  # stores grad_output and index
        self.prev_grad_Q = {}
        self.prev_grad_K = {}
        self.prev_grad_V = {}
        self.step_counter = {}

    def forward(self, hidden_states):
        device = hidden_states.device
        self.step_counter.setdefault(device, 0)
        self.prev_gradients.setdefault(device, None)
        self.prev_grad_Q.setdefault(device, None)
        self.prev_grad_K.setdefault(device, None)
        self.prev_grad_V.setdefault(device, None)

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Decide whether to recompute
        step = self.step_counter[device]
        reuse_percentage = get_current_reuse_percentage(self.reuse_schedule, step)
        prev_reuse_percentage = get_current_reuse_percentage(self.reuse_schedule, step - 1)
        if reuse_percentage != prev_reuse_percentage and (device.index is None or device.index == 0):
            print('Switching REUSE_PERCENTAGE from', prev_reuse_percentage, 'to', reuse_percentage)
        should_recompute = (step % self.k == 0)

        if should_recompute and self.prev_gradients[device] is not None:
            with torch.no_grad():
                grad_output = self.prev_gradients[device]["grad_output"]
                idx = self.prev_gradients[device]["index"]

                V_slice = V[idx:idx+1]
                Q_slice = Q[idx:idx+1]
                K_slice = K[idx:idx+1]

                d_k = Q.size(-1)
                attn_scores = torch.bmm(Q_slice, K_slice.transpose(1, 2)) / (d_k ** 0.5)
                attn_probs = torch.softmax(attn_scores, dim=-1)

                grad_attn_probs = torch.bmm(grad_output, V_slice.transpose(1, 2))
                grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_output)

                grad_attn_scores = grad_attn_probs * attn_probs
                grad_attn_scores -= attn_probs * grad_attn_scores.sum(dim=-1, keepdim=True)
                grad_attn_scores /= (d_k ** 0.5)

                grad_Q = torch.bmm(grad_attn_scores, K_slice)
                grad_K = torch.bmm(grad_attn_scores.transpose(1, 2), Q_slice)

                self.prev_grad_Q[device] = grad_Q.detach()
                self.prev_grad_K[device] = grad_K.detach()
                self.prev_grad_V[device] = grad_V.detach()

        output = ReSpropAttentionFunction.apply(
            Q, K, V,
            self.prev_grad_Q[device],
            self.prev_grad_K[device],
            self.prev_grad_V[device],
            reuse_percentage
        )

        if output.requires_grad:
            def hook(grad_output):
                self.step_counter[device] += 1
                if self.step_counter[device] % self.k == 0:
                    rand_idx = torch.randint(0, grad_output.size(0), (1,)).item()
                    sampled_grad = grad_output[rand_idx:rand_idx+1].clone().detach()
                    self.prev_gradients[device] = {
                        "grad_output": sampled_grad,
                        "index": rand_idx
                    }

            output.register_hook(hook)
        else:
            self.prev_gradients[device] = None

        return output


    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, reuse_percentage={self.reuse_percentage}"


def respropify_bert_att_k(base_model, att_reuse_schedule=None, lin_reuse_schedule=None, lin_k=1, att_k=1):
    model = copy.deepcopy(base_model).to(torch.cuda.current_device())

    def resprop_attention_block(embed_dim):
        return ReSpropAttention(embed_dim, reuse_schedule=att_reuse_schedule, lin_reuse_schedule= lin_reuse_schedule, att_k=att_k, lin_k=lin_k)

    def resprop_linear(layer: nn.Linear):
        return ReSpropLinear(layer.in_features, layer.out_features, layer.bias is not None, reuse_schedule=lin_reuse_schedule, k=lin_k)

    for layer in model.bert.encoder.layer:
        att = layer.attention

        embed_dim = att.self.query.in_features
        att.self.custom_attention = resprop_attention_block(embed_dim)

        if hasattr(att.self, "query"):
            del att.self.query
        if hasattr(att.self, "key"):
            del att.self.key
        if hasattr(att.self, "value"):
            del att.self.value

        layer.intermediate.dense = resprop_linear(layer.intermediate.dense)
        layer.output.dense = resprop_linear(layer.output.dense)

    return model



def patch_bert_self_attention_k(model):
    for layer in model.bert.encoder.layer:
        attn_self = layer.attention.self

        if not hasattr(attn_self, "_original_forward"):
            attn_self._original_forward = attn_self.forward

        def new_forward(self, hidden_states, attention_mask=None, head_mask=None,
                        encoder_hidden_states=None, encoder_attention_mask=None,
                        past_key_value=None, output_attentions=False):
            if hasattr(self, "custom_attention"):
                hidden_states = self.custom_attention(hidden_states)

                attention_output = hidden_states
                outputs = (attention_output,)

                if output_attentions:
                    dummy_attn_probs = torch.zeros(
                        hidden_states.size(0),
                        hidden_states.size(1),
                        hidden_states.size(1),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    outputs += (dummy_attn_probs,)

                return outputs
            else:
                return self._original_forward(
                    hidden_states, attention_mask, head_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    past_key_value, output_attentions
                )

        attn_self.forward = types.MethodType(new_forward, attn_self)