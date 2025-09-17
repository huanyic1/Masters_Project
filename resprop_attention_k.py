import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import types

import torch.nn.functional as F


def compare_parameters(param1, param2, name=""):
    cos_sim = F.cosine_similarity(param1.view(-1), param2.view(-1), dim=0).item()
    max_diff = (param1 - param2).abs().max().item()
    print(f"{name}:")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  Max absolute difference: {max_diff:.6e}")

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
        k_keep = int(max(0, math.floor((1-reuse_percentage) * N)))
        if k_keep <= 0:
            return torch.zeros_like(grad_diff)
        if k_keep >= N:
            return grad_diff

        abs_flat = grad_diff.abs().reshape(-1)
        vals, idx = torch.topk(abs_flat, k_keep, largest=True, sorted=False)

        out = torch.zeros_like(grad_diff).reshape(-1)
        out.scatter_(0, idx, grad_diff.reshape(-1).index_select(0, idx))

        return out.view_as(grad_diff)


class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, reuse_percentage, structured=False, n=2, group_size=4):
        prev_grad_output = prev_grad_output if prev_grad_output is not None else None

        if prev_grad_output is not None and reuse_percentage > 0 and len(input.shape) == 3 and prev_grad_output.size(0) == input.size(1):
            with torch.no_grad():
                # Precompute prev_grad_input and prev_grad_weight
                # prev_grad_input = torch.mm(prev_grad_output, weight)                        # pre∇w_l
                # prev_grad_weight = torch.mm(prev_grad_output.t(), torch.sum(input, dim=0))
                prev_grad_input = prev_grad_output.matmul(weight).detach()              # [T, I]
                sum_input = input.sum(dim=0)                                   # [T, I]
                prev_grad_weight = prev_grad_output.t().matmul(sum_input).detach()  # [I, O]
        else:
            if prev_grad_output is not None:
                print("Warning: Couldn't reuse gradient due to shape mis-match.")
            prev_grad_output = prev_grad_input = prev_grad_weight = None

        ctx.reuse_percentage = reuse_percentage
        ctx.structured = structured
        ctx.n = n
        ctx.group_size = group_size
        ctx.prev_grad_output = prev_grad_output
        ctx.prev_grad_input = prev_grad_input
        ctx.prev_grad_weight = prev_grad_weight
        ctx.save_for_backward(input, weight, bias)

        B, T, I = input.shape
        out = F.linear(input.view(-1, I), weight, bias)  # [(B*T), O]
        return out.view(B, T, weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        prev_grad_output, prev_grad_input, prev_grad_weight = ctx.prev_grad_output, ctx.prev_grad_input, ctx.prev_grad_weight

        grad_input = grad_weight = grad_bias = None

        # Compute reuse mask
        if prev_grad_output is not None:
            grad_diff = generate_reuse_mask(ctx.reuse_percentage, grad_output, prev_grad_output, ctx.structured, ctx.n, ctx.group_size)
            grad_output = grad_diff
            print(grad_diff)
        
        # Compute gradients
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
            # grad_input = torch.bmm(grad_output, weight.expand(grad_output.size(0), -1, -1))
            if prev_grad_output is not None:
                grad_input = grad_input.add(prev_grad_input.unsqueeze(0))
                # grad_input += prev_grad_input


        if ctx.needs_input_grad[1]:
            # grad_weight = torch.sum(torch.bmm(grad_output.transpose(1, 2), input), dim=0)
            grad_weight = torch.einsum('bto,bti->oi', grad_output, input)
            if prev_grad_output is not None:
                grad_weight = grad_weight.add(prev_grad_weight)
                # grad_weight += prev_grad_weight

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class ReSpropLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, reuse_schedule=None, k=1, avg=True):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.reuse_schedule = reuse_schedule or [(0.9, 0)]
        self.k = k
        self.prev_gradients = {}
        self.step_counter = {}
        self.avg = avg

    def forward(self, input):
        device = input.device
        self.prev_gradients.setdefault(device, None)
        self.step_counter.setdefault(device, 0)
        step = self.step_counter[device]
        num1, num2, structured = get_current_reuse_percentage(self.reuse_schedule, step)
        if not structured:
            reuse_percentage = num1
            n = None
            group_size = None
        else:
            reuse_percentage = num1/num2
            n = num1
            group_size = num2

        output = ReSpropLinearFunction.apply(
            input, self.weight,
            self.bias if self.bias is not None else None,
            self.prev_gradients[device],
            reuse_percentage, 
            structured,
            n,
            group_size
        )

        if output.requires_grad:
            def hook(grad_output):
                if reuse_percentage > 0: #and self.step_counter[device] % self.k == 0:
                    if self.avg:
                        self.prev_gradients[device] =  grad_output.sum(dim=0) / grad_output.size(0) # torch.mean(grad_output, dim=0) #
                    else: 
                        self.prev_gradients[device] = grad_output[torch.randint(0, grad_output.size(0), (1,))][0].clone().detach()
                self.step_counter[device] += 1
                
            output.register_hook(hook)

        return output

    def extra_repr(self):
        return super().extra_repr() + f", reuse_percentage={self.reuse_percentage}"

        
class ReSpropAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, prev_grad_output, prev_attn_prod, prev_V_prod, prev_soft_grad, prev_K_prod, prev_Q_prod, reuse_percentage, structured=False, n=2, group_size=4):
        """
        prev_grad_output = dL/dZ_prev
        prev_attn_prod = A.T * dL/dZ_prev
        prev_V_prod = dL/dZ_prev * V.T
        prev_soft_grad = dL/dSoftmax_prev
        prev_K_prod = dL/dSoftmax_prev * K
        prev_Q_prod = dL/dSoftmax_prev.T * Q
        """
        ctx.reuse_percentage = reuse_percentage
        ctx.structured = structured
        ctx.n = n
        ctx.group_size = group_size
        ctx.prev_grad_output = None if prev_grad_output is None else prev_grad_output.detach()
        ctx.prev_attn_prod  = None if prev_attn_prod  is None else prev_attn_prod.detach()
        ctx.prev_V_prod     = None if prev_V_prod     is None else prev_V_prod.detach()
        ctx.prev_soft_grad  = None if prev_soft_grad  is None else prev_soft_grad.detach()
        ctx.prev_K_prod     = None if prev_K_prod     is None else prev_K_prod.detach()
        ctx.prev_Q_prod     = None if prev_Q_prod     is None else prev_Q_prod.detach()

        ctx.save_for_backward(Q, K, V)

        # d_k = Q.size(-1)
        # attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        # attn_probs = torch.softmax(attn_scores, dim=-1)
        # ctx.attn_probs = attn_probs  # current softmax

        # return torch.bmm(attn_probs, V)
        
        d_k = Q.size(-1)
        # attn_scores = Q @ K^T / sqrt(d_k)
        # attn_scores = torch.matmul(Q, K.transpose(1, 2)) * (d_k ** -0.5)
        attn_scores = torch.einsum('btd, bud -> btu', Q, K) * (d_k ** -0.5)
        attn_probs = attn_scores.softmax(dim=-1)
        # Save detached probs (we only need numbers, not gradients through probs)
        ctx.attn_probs = attn_probs.detach()

        # out = A @ V
        return torch.einsum('btu, buv -> btv', attn_probs, V)
        # return torch.matmul(attn_probs, V)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        prev_grad_output, prev_attn_prod, prev_V_prod, prev_soft_grad, prev_K_prod, prev_Q_prod = ctx.prev_grad_output, ctx.prev_attn_prod, ctx.prev_V_prod, ctx.prev_soft_grad, ctx.prev_K_prod, ctx.prev_Q_prod

        
        attn_probs = ctx.attn_probs
        d_k = Q.size(-1)
        grad_Q=grad_K=grad_V=None

        if prev_grad_output is not None and ctx.reuse_percentage > 0:
            grad_diff = generate_reuse_mask(ctx.reuse_percentage, grad_output, prev_grad_output, ctx.structured, ctx.n, ctx.group_size)
            # grad_attn_probs = torch.bmm(grad_diff, V.transpose(1, 2)) + prev_V_prod
            # grad_attn_probs = torch.matmul(grad_diff, V.transpose(1, 2)) + prev_V_prod
            grad_attn_probs = torch.einsum('btv, buv -> btu', grad_diff, V) + prev_V_prod
            if ctx.needs_input_grad[2]:
                # grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_diff) + prev_attn_prod
                grad_V = torch.einsum('btu,btv->buv', attn_probs, grad_diff) + prev_attn_prod
                # grad_V = torch.matmul(attn_probs.transpose(1, 2), grad_diff) + prev_attn_prod
        else: 
            # grad_attn_probs = torch.bmm(grad_output, V.transpose(1, 2))
            # grad_attn_probs = torch.matmul(grad_output, V.transpose(1, 2))
            grad_attn_probs = torch.einsum('btv, buv -> btu', grad_output, V)
            if ctx.needs_input_grad[2]:
                # grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_output)
                # grad_V = torch.matmul(attn_probs.transpose(1, 2), grad_output)
                grad_V = torch.einsum('btu,btv->buv', attn_probs, grad_output)

        #grad_attn_scores = torch.einsum('btu, btu -> bt', grad_attn_probs, attn_probs)
        #grad_attn_scores = (grad_attn_probs - attn_probs * grad_attn_scores.unsqueeze(-1)) * attn_probs
        #grad_attn_scores = grad_attn_scores * (d_k ** -0.5)
        # Rowwise dot: sum over last dim
        row_dot = (grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True)  # [B, T, 1]
        grad_attn_scores = (grad_attn_probs - row_dot) * attn_probs         # [B, T, T]
        grad_attn_scores = grad_attn_scores * (d_k ** -0.5)


        # grad_attn_scores = grad_attn_probs * attn_probs
        # grad_attn_scores -= attn_probs * grad_attn_scores.sum(dim=-1, keepdim=True)
        # grad_attn_scores *= (d_k ** -0.5)

        if prev_soft_grad is not None and ctx.reuse_percentage > 0:
            grad_attn_diff = generate_reuse_mask(ctx.reuse_percentage, grad_attn_scores, prev_soft_grad, ctx.structured, ctx.n, ctx.group_size)
            if ctx.needs_input_grad[0]:
                # grad_Q = torch.matmul(grad_attn_diff, K) + prev_K_prod
                # grad_Q = torch.bmm(grad_attn_diff, K) + prev_K_prod
                grad_Q = torch.einsum('btu, bud -> btd', grad_attn_diff, K) + prev_K_prod
            if ctx.needs_input_grad[1]:
                # grad_K = torch.bmm(grad_attn_diff.transpose(1, 2), Q) + prev_Q_prod
                # grad_K = torch.matmul(grad_attn_diff.transpose(1, 2), Q) + prev_Q_prod
                grad_K = torch.einsum('btu, btd -> bud', grad_attn_diff, Q) + prev_Q_prod
        else:
            if ctx.needs_input_grad[0]:
                # grad_Q = torch.bmm(grad_attn_scores, K)
                # grad_Q = torch.matmul(grad_attn_scores, K)
                grad_Q = torch.einsum('btu, bud -> btd', grad_attn_scores, K)
            if ctx.needs_input_grad[1]:
                # grad_K = torch.bmm(grad_attn_scores.transpose(1, 2), Q)
                # grad_K = torch.matmul(grad_attn_scores.transpose(1, 2), Q)
                grad_K = torch.einsum('btu, btd -> bud', grad_attn_scores, Q)

        return grad_Q, grad_K, grad_V, None, None, None, None, None, None, None, None, None, None
   
def resprop_linear(layer: nn.Linear, reuse_schedule=None, k=1):
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

class ReSpropAttention(nn.Module):
    def __init__(self, att, reuse_schedule=None, lin_reuse_schedule=None, att_k=1, lin_k=1):
        super().__init__()
        embed_dim = att.query.in_features
        self.embed_dim = embed_dim
        self.reuse_schedule = reuse_schedule or [(0.9, 0)]
        self.k = att_k

        if lin_reuse_schedule:
            self.q_proj = resprop_linear(att.query, reuse_schedule=lin_reuse_schedule, k=lin_k)
            self.k_proj = resprop_linear(att.key, reuse_schedule=lin_reuse_schedule, k=lin_k)
            self.v_proj = resprop_linear(att.value, reuse_schedule=lin_reuse_schedule, k=lin_k)
        else:
            self.q_proj = att.query
            self.k_proj = att.key
            self.v_proj = att.value
        self.prev_grad_output = {}  # stores grad_output and index
        self.prev_attn_prods = {}
        self.prev_V_prods = {}
        self.prev_soft_grads = {}
        self.prev_K_prods = {}
        self.prev_Q_prods = {}
        self.step_counter = {}

    def forward(self, hidden_states):
        device = hidden_states.device
        self.step_counter.setdefault(device, 0)
        self.prev_grad_output.setdefault(device, None)
        self.prev_attn_prods.setdefault(device, None)
        self.prev_V_prods.setdefault(device, None)
        self.prev_soft_grads.setdefault(device, None)
        self.prev_K_prods.setdefault(device, None)
        self.prev_Q_prods.setdefault(device, None)

        num1, num2, structured= get_current_reuse_percentage(self.reuse_schedule, self.step_counter[device])
        if not structured:
            reuse_percentage = num1
            n = None
            group_size = None
        else:
            reuse_percentage = num1/num2
            n = num1
            group_size = num2
        
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        output = ReSpropAttentionFunction.apply(
            Q, K, V,
            self.prev_grad_output[device],
            self.prev_attn_prods[device],
            self.prev_V_prods[device],
            self.prev_soft_grads[device],
            self.prev_K_prods[device],
            self.prev_Q_prods[device],
            reuse_percentage, 
            structured,
            n,
            group_size
        )
        if output.requires_grad:
            def hook(grad_output):
                if reuse_percentage>0 and self.step_counter[device] % self.k == 0:
                    with torch.no_grad():
                        # sampled_grad = grad_output.sum(dim=0, keepdim=True)/grad_output.size(0) #grad_output.mean(dim=0, keepdim=True).detach()  # [1, T, d] 

                        # Q_ = Q.sum(dim=0, keepdim=True)/Q.size(0) #Q.mean(dim=0, keepdim=True)  # [1, T, d_k]
                        # K_ = K.sum(dim=0, keepdim=True)/K.size(0) #K.mean(dim=0, keepdim=True)  # [1, T, d_k]
                        # V_ = V.sum(dim=0, keepdim=True)/V.size(0) #V.mean(dim=0, keepdim=True)  # [1, T, d_v]
                        # d_k = Q.size(-1)

                        # attn_scores = torch.bmm(Q_, K_.transpose(1, 2)) / (d_k ** 0.5)  # [1, T, T]
                        # attn_probs = torch.softmax(attn_scores, dim=-1)                 # [1, T, T]

                        # attn_prod = torch.bmm(attn_probs.transpose(1, 2), sampled_grad)  # [1, T, d]
                        # V_prod = torch.bmm(sampled_grad, V_.transpose(1, 2))             # [1, d, T]

                        # grad_softmax = torch.bmm(sampled_grad, V_.transpose(1, 2))       # [1, T, T]
                        # grad_attn_scores = grad_softmax * attn_probs                     # [1, T, T]
                        # grad_attn_scores -= attn_probs * grad_attn_scores.sum(dim=-1, keepdim=True)
                        # grad_attn_scores /= (d_k ** 0.5)

                        # K_prod = torch.bmm(grad_attn_scores, K_)                         # [1, T, d]
                        # Q_prod = torch.bmm(grad_attn_scores.transpose(1, 2), Q_)         # [1, T, d]

                        # self.prev_grad_output[device] = sampled_grad
                        # self.prev_attn_prods[device] = attn_prod.detach()
                        # self.prev_V_prods[device] = V_prod.detach()
                        # self.prev_soft_grads[device] = grad_attn_scores.detach()
                        # self.prev_K_prods[device] = K_prod.detach()
                        # self.prev_Q_prods[device] = Q_prod.detach()

                        sampled_grad = grad_output.mean(dim=0, keepdim=True).detach()   # [1, T, d]
                        Q_ = Q.mean(dim=0, keepdim=True)                                 # [1, T, d_k]
                        K_ = K.mean(dim=0, keepdim=True)                                 # [1, T, d_k]
                        V_ = V.mean(dim=0, keepdim=True)                                 # [1, T, d_v]

                        d_k = Q.size(-1)
                        inv_sqrt_dk = 1.0 / math.sqrt(d_k)

                        # Attn probs: A = softmax((Q K^T) / sqrt(d_k)) → [1, T, T]
                        attn_scores = torch.einsum('btd,bud->btu', Q_, K_) * inv_sqrt_dk
                        attn_probs  = attn_scores.softmax(dim=-1)                         # [1, T, T]

                        # dL/dV: A^T @ sampled_grad → [1, T, d]
                        attn_prod = torch.einsum('btu,btd->bud', attn_probs, sampled_grad)  # uses A^T implicitly via indices

                        # G = dL/dA = sampled_grad @ V^T → [1, T, T]
                        V_prod = torch.einsum('btd,bdu->btu', sampled_grad, V_.transpose(1, 2))

                        # softmax backward to scores: grad_scores = (G - A * sum(G*A)) * A / sqrt(d_k)
                        s = torch.einsum('btu,btu->bt', V_prod, attn_probs).unsqueeze(-1)    # [1, T, 1]
                        grad_attn_scores = (V_prod - attn_probs * s) * attn_probs * inv_sqrt_dk  # [1, T, T]

                        # K/Q products for reuse
                        K_prod = torch.einsum('btu,bud->btd', grad_attn_scores, K_)        # [1, T, d_k]
                        Q_prod = torch.einsum('but,btd->bud', grad_attn_scores, Q_)        # [1, T, d_k]

                        # Cache (detached)
                        self.prev_grad_output[device] = sampled_grad
                        self.prev_attn_prods[device]  = attn_prod.detach()
                        self.prev_V_prods[device]     = V_prod.detach()            # (same as old grad_softmax)
                        self.prev_soft_grads[device]  = grad_attn_scores.detach()
                        self.prev_K_prods[device]     = K_prod.detach()
                        self.prev_Q_prods[device]     = Q_prod.detach()

                self.step_counter[device] += 1

            output.register_hook(hook)

        return output


    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, reuse_percentage={self.reuse_percentage}"


def respropify_bert_att_k(base_model, att_reuse_schedule=None, lin_reuse_schedule=None, lin_k=1, att_k=1):
    model = copy.deepcopy(base_model).to(torch.cuda.current_device())

    def resprop_attention_block(att):
        return ReSpropAttention(att, reuse_schedule=att_reuse_schedule, lin_reuse_schedule= lin_reuse_schedule, att_k=att_k, lin_k=lin_k)

    def resprop_linear(layer: nn.Linear):
        new_layer = ReSpropLinear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            reuse_schedule=lin_reuse_schedule
        )
        new_layer.weight.data.copy_(layer.weight.data)
        if layer.weight.grad is not None:
            new_layer.weight.grad.data.copy_(layer.weight.grad.data)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
            if layer.bias.grad is not None:
                new_layer.bias.grad.data.copy_(layer.bias.grad.data)
        return new_layer
    for layer in model.bert.encoder.layer:
        att = layer.attention
        att.self.custom_attention = resprop_attention_block(att.self)
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


def main(): 
    pass

if __name__ == "__main__":
    main()