import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import types
#!/usr/bin/env python3
import os
import argparse
from datasets import load_from_disk, concatenate_datasets, load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification
from copy import deepcopy


def cos_sim(a, b, eps=1e-12):
    """Cosine similarity with float64 precision and epsilon for numerical stability.

    Uses float64 to avoid precision loss and adds epsilon to prevent division by zero.
    """
    a_flat = a.view(-1).to(torch.float64)
    b_flat = b.view(-1).to(torch.float64)

    # Compute cosine similarity manually with epsilon for stability
    dot_product = (a_flat * b_flat).sum()
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()

    cos_val = (dot_product / (a_norm * b_norm + eps)).item()


    return cos_val

def cos_sim_second_moment(a, b, eps=1e-30):
    """Cosine similarity of second moments (squared values) with numerically stable computation.

    Problem: Computing dot(a^2, b^2) involves 4th powers which causes underflow for small values.

    Solution: Use log-space arithmetic to avoid underflow:
    cos(a^2, b^2) = exp(log(dot(a^2, b^2)) - 0.5*log(sum(a^4)) - 0.5*log(sum(b^4)))

    We compute everything in log space, then exponentiate at the end.
    """
    # Check if a and b are actually the same object
    same_object = a is b

    a_flat = a.view(-1).to(torch.float64)
    b_flat = b.view(-1).to(torch.float64) if not same_object else a_flat

    # For same tensor, return 1.0 immediately
    if same_object:
        return 1.0

    # Take absolute values to work in log space
    a_abs = a_flat.abs()
    b_abs = b_flat.abs()

    # Compute log of squared values: log(a^2) = 2*log(|a|)
    # Add eps to avoid log(0)
    log_a_sq = 2.0 * torch.log(a_abs + eps)
    log_b_sq = 2.0 * torch.log(b_abs + eps)

    # For the dot product: dot(a^2, b^2) = sum(a^2 * b^2)
    # In log space: log(sum(exp(log(a^2) + log(b^2))))
    # This is the logsumexp operation
    log_ab_sq = log_a_sq + log_b_sq  # Element-wise
    log_dot = torch.logsumexp(log_ab_sq, dim=0)

    # For the norms: norm(a^2)^2 = sum(a^4)
    # log(sum(a^4)) = log(sum(exp(4*log(|a|))))
    log_a_4 = 4.0 * torch.log(a_abs + eps)
    log_b_4 = 4.0 * torch.log(b_abs + eps)

    log_sum_a4 = torch.logsumexp(log_a_4, dim=0)
    log_sum_b4 = torch.logsumexp(log_b_4, dim=0)

    # norm(a^2) = sqrt(sum(a^4)) => log(norm(a^2)) = 0.5 * log(sum(a^4))
    log_norm_a_sq = 0.5 * log_sum_a4
    log_norm_b_sq = 0.5 * log_sum_b4

    # Cosine = dot / (norm_a * norm_b)
    # log(cosine) = log(dot) - log(norm_a) - log(norm_b)
    log_cos = log_dot - log_norm_a_sq - log_norm_b_sq

    # Convert back from log space
    cos_val = torch.exp(log_cos).item()


    return cos_val

def cos_sim_batch(a, b, eps=1e-8):
    """Mean cosine similarity over batch dimension."""
    if a is None or b is None:
        return 0.0
    B = a.size(0)
    a_flat = a.view(B, -1)
    b_flat = b.view(1, -1).expand(B, -1)
    a_norm = a_flat / (a_flat.norm(dim=-1, keepdim=True) + eps)
    b_norm = b_flat / (b_flat.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1).mean().item()

cosine_logs = {
    "lin_grad_inputs": 0.0,
    "lin_grad_weights": 0.0,
    "lin_v2_inputs": 0.0,
    "lin_v2_weights": 0.0,
    "lin_count": 0,
    "att_grad_Q": 0.0,
    "att_grad_K": 0.0,
    "att_grad_V": 0.0,
    "att_grad_probs": 0.0,
    "att_v2_Q": 0.0,
    "att_v2_K": 0.0,
    "att_v2_V": 0.0,
    "att_count": 0,
}

def print_cosine_logs():
    """Aggregate printout every N steps."""
    if cosine_logs["lin_count"] > 0:
        print(f"[Linear] grad_input cos={cosine_logs['lin_grad_inputs']/cosine_logs['lin_count']:.4f}  "
              f"grad_weight cos={cosine_logs['lin_grad_weights']/cosine_logs['lin_count']:.4f}  "
              f"v2_input cos={cosine_logs['lin_v2_inputs']/cosine_logs['lin_count']:.4f}  "
              f"v2_weight cos={cosine_logs['lin_v2_weights']/cosine_logs['lin_count']:.4f}")
    if cosine_logs["att_count"] > 0:
        print(f"[Attention] grad_Q={cosine_logs['att_grad_Q']/cosine_logs['att_count']:.4f}  "
              f"grad_K={cosine_logs['att_grad_K']/cosine_logs['att_count']:.4f}  "
              f"grad_V={cosine_logs['att_grad_V']/cosine_logs['att_count']:.4f}  "
              f"v2_Q={cosine_logs['att_v2_Q']/cosine_logs['att_count']:.4f}  "
              f"v2_K={cosine_logs['att_v2_K']/cosine_logs['att_count']:.4f}  "
              f"v2_V={cosine_logs['att_v2_V']/cosine_logs['att_count']:.4f}")


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


def correct_grad_norm(grad_output, prev_grad_output, reuse_grad):
    """
    grad_output: the current dense gradient (B,T,D)
    prev_grad_output: the reused dense gradient (T,D)
    reuse_grad: the hybrid gradient = (masked diff) + prev_grad_output
    """
    dense_norm = grad_output.norm(p=2)
    reuse_norm = reuse_grad.norm(p=2).clamp(min=1e-8)
    scale = (dense_norm / reuse_norm).detach()
    return reuse_grad * scale


class ReSpropLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, prev_grad_output, reuse_percentage, structured=False, n=2, group_size=4, step=0):
        prev_grad_output = prev_grad_output if prev_grad_output is not None else None

        if prev_grad_output is not None and len(input.shape) == 3 and prev_grad_output.size(0) == input.size(1):
            with torch.no_grad():
                prev_grad_input = prev_grad_output.matmul(weight).detach()              # [T, I]
                sum_input = input.sum(dim=0)                                   # [T, I]
                prev_grad_weight = prev_grad_output.t().matmul(sum_input).detach()  # [I, O]
        else:
            if prev_grad_output is not None and reuse_percentage > 0:
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
        ctx.step = step

        B, T, I = input.shape
        out = F.linear(input.view(-1, I), weight, bias)  # [(B*T), O]
        return out.view(B, T, weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        prev_grad_output, prev_grad_input, prev_grad_weight = ctx.prev_grad_output, ctx.prev_grad_input, ctx.prev_grad_weight

        og_grad_output = grad_output
        grad_input = grad_weight = grad_bias = None

        # Compute reuse mask
        if prev_grad_output is not None:
            grad_diff = generate_reuse_mask(ctx.reuse_percentage, grad_output, prev_grad_output, ctx.structured, ctx.n, ctx.group_size)
            grad_output = grad_diff
        
        # Compute gradients
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
            if prev_grad_output is not None:
                grad_input = grad_input.add(prev_grad_input.unsqueeze(0))
                #grad_input = correct_grad_norm(grad_output, prev_grad_output, grad_input)


        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('bto,bti->oi', grad_output, input)
            if prev_grad_output is not None:
                grad_weight = grad_weight.add(prev_grad_weight)
                #grad_weight = correct_grad_norm(grad_output, prev_grad_output, grad_weight)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 1))

        # --- Logging for analysis ---
        if prev_grad_output is not None:
            # "True" reference (no reuse) gradients
            og_grad_input = torch.matmul(og_grad_output, weight)
            og_grad_weight = torch.einsum('bto,bti->oi', og_grad_output, input)

            cosine_logs['lin_grad_inputs'] += cos_sim(og_grad_input, grad_input)
            cosine_logs['lin_grad_weights'] += cos_sim(og_grad_weight, grad_weight)

            # Second-moment cosine: compare g^2 terms (use numerically stable version)
            cosine_logs['lin_v2_inputs'] += cos_sim_second_moment(og_grad_input, grad_input)
            cosine_logs['lin_v2_weights'] += cos_sim_second_moment(og_grad_weight, grad_weight)
            cosine_logs['lin_count'] += 1

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

        # if step % 250 == 0: 
        #     reuse_percentage = 0

        output = ReSpropLinearFunction.apply(
            input, self.weight,
            self.bias if self.bias is not None else None,
            self.prev_gradients[device],
            reuse_percentage, 
            structured,
            n,
            group_size, 
            step
        )

        if output.requires_grad:
            def hook(grad_output):
                # if reuse_percentage > 0: #and self.step_counter[device] % self.k == 0:
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
        
        d_k = Q.size(-1)
        attn_scores = torch.einsum('btd, bud -> btu', Q, K) * (d_k ** -0.5)
        attn_probs = attn_scores.softmax(dim=-1)

        ctx.attn_probs = attn_probs.detach()

        return torch.einsum('btu, buv -> btv', attn_probs, V)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        prev_grad_output, prev_attn_prod, prev_V_prod, prev_soft_grad, prev_K_prod, prev_Q_prod = ctx.prev_grad_output, ctx.prev_attn_prod, ctx.prev_V_prod, ctx.prev_soft_grad, ctx.prev_K_prod, ctx.prev_Q_prod

        
        attn_probs = ctx.attn_probs
        d_k = Q.size(-1)
        grad_Q=grad_K=grad_V=None

        if prev_grad_output is not None and ctx.reuse_percentage > 0:
            grad_diff = generate_reuse_mask(ctx.reuse_percentage, grad_output, prev_grad_output, ctx.structured, ctx.n, ctx.group_size)
            grad_attn_probs = torch.einsum('btv, buv -> btu', grad_diff, V) + prev_V_prod
            if ctx.needs_input_grad[2]:
                grad_V = torch.einsum('btu,btv->buv', attn_probs, grad_diff) + prev_attn_prod
                #grad_V = correct_grad_norm(grad_output, prev_grad_output, grad_V)
        else: 
            grad_attn_probs = torch.einsum('btv, buv -> btu', grad_output, V)
            if ctx.needs_input_grad[2]:
                grad_V = torch.einsum('btu,btv->buv', attn_probs, grad_output)

        row_dot = (grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True)  # [B, T, 1]
        grad_attn_scores = (grad_attn_probs - row_dot) * attn_probs         # [B, T, T]
        grad_attn_scores = grad_attn_scores * (d_k ** -0.5)

        if prev_soft_grad is not None and ctx.reuse_percentage > 0:
            grad_attn_diff = generate_reuse_mask(ctx.reuse_percentage, grad_attn_scores, prev_soft_grad, ctx.structured, ctx.n, ctx.group_size)
            if ctx.needs_input_grad[0]:
                grad_Q = torch.einsum('btu, bud -> btd', grad_attn_diff, K) + prev_K_prod
                #grad_Q = correct_grad_norm(grad_attn_scores, prev_soft_grad, grad_Q)
            if ctx.needs_input_grad[1]:
                grad_K = torch.einsum('btu, btd -> bud', grad_attn_diff, Q) + prev_Q_prod
                #grad_K = correct_grad_norm(grad_attn_scores, prev_soft_grad, grad_K)
        else:
            if ctx.needs_input_grad[0]:
                grad_Q = torch.einsum('btu, bud -> btd', grad_attn_scores, K)
            if ctx.needs_input_grad[1]:
                grad_K = torch.einsum('btu, btd -> bud', grad_attn_scores, Q)

        if prev_grad_output is not None:
            # True baseline (dense)
            d_k = Q.size(-1)
            attn_probs = ctx.attn_probs
            og_grad_attn_probs = torch.einsum('btv, buv -> btu', grad_output, V)
            og_grad_attn_scores = (og_grad_attn_probs - (og_grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True)) * attn_probs * (d_k ** -0.5)
            og_grad_Q = torch.einsum('btu,bud->btd', og_grad_attn_scores, K)
            og_grad_K = torch.einsum('btu,btd->bud', og_grad_attn_scores, Q)
            og_grad_V = torch.einsum('btu,btv->buv', attn_probs, grad_output)


            cosine_logs['att_grad_Q'] += cos_sim(og_grad_Q, grad_Q)
            cosine_logs['att_grad_K'] += cos_sim(og_grad_K, grad_K)
            cosine_logs['att_grad_V'] += cos_sim(og_grad_V, grad_V)
            cosine_logs['att_grad_probs'] += cos_sim(og_grad_attn_probs, grad_attn_probs)

            # Second-moment cosine (squared gradients) - use numerically stable version
            cosine_logs['att_v2_Q'] += cos_sim_second_moment(og_grad_Q, grad_Q)
            cosine_logs['att_v2_K'] += cos_sim_second_moment(og_grad_K, grad_K)
            cosine_logs['att_v2_V'] += cos_sim_second_moment(og_grad_V, grad_V)
            cosine_logs['att_count'] += 1

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

        # if self.step_counter[device] % 250 == 0: 
        #     reuse_percentage = 0
        
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


                        grad_attn_probs = torch.einsum('btv, buv -> btu', sampled_grad, V_)
                        row_dot = (grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True)  
                        grad_attn_scores = (grad_attn_probs - row_dot) * attn_probs    
                        grad_attn_scores = grad_attn_scores * inv_sqrt_dk


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

def _normalize_omitted(layers, omitted_layers):
    """
    Convert omitted_layers into a set of valid non-negative indices.
    Handles negative indices like Python lists.
    """
    n = len(layers)
    if omitted_layers is None:
        return {0, n - 1}
    result = set()
    for idx in omitted_layers:
        if idx < 0:
            idx = n + idx  # convert negative index
        if 0 <= idx < n:
            result.add(idx)
    return result

def respropify_bert_att_k(
    base_model,
    att_reuse_schedule=None,
    lin_reuse_schedule=None,
    lin_k=1,
    att_k=1,
    omitted_layers=None,
):
    model = copy.deepcopy(base_model).to(torch.cuda.current_device())
    layers = model.bert.encoder.layer
    omitted_layers = _normalize_omitted(layers, omitted_layers)

    def resprop_attention_block(att):
        return ReSpropAttention(
            att,
            reuse_schedule=att_reuse_schedule,
            lin_reuse_schedule=lin_reuse_schedule,
            att_k=att_k,
            lin_k=lin_k,
        )

    def resprop_linear(layer: nn.Linear):
        new_layer = ReSpropLinear(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            reuse_schedule=lin_reuse_schedule,
        )
        new_layer.weight.data.copy_(layer.weight.data)
        if layer.weight.grad is not None:
            new_layer.weight.grad.data.copy_(layer.weight.grad.data)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
            if layer.bias.grad is not None:
                new_layer.bias.grad.data.copy_(layer.bias.grad.data)
        return new_layer

    layers = model.bert.encoder.layer
    for i, layer in enumerate(layers):
        if i in omitted_layers:
            print(f"Leaving layer {i} untouched")
            continue

        att = layer.attention
        att.self.custom_attention = resprop_attention_block(att.self)
        layer.intermediate.dense = resprop_linear(layer.intermediate.dense)
        layer.output.dense = resprop_linear(layer.output.dense)

    return model


def patch_bert_self_attention_k(model, omitted_layers=None):
    layers = model.bert.encoder.layer
    omitted_layers = _normalize_omitted(layers, omitted_layers)

    for i, layer in enumerate(layers):
        if i in omitted_layers:
            continue

        attn_self = layer.attention.self

        if not hasattr(attn_self, "_original_forward"):
            attn_self._original_forward = attn_self.forward

        def new_forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        ):
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
                        dtype=hidden_states.dtype,
                    )
                    outputs += (dummy_attn_probs,)

                return outputs
            else:
                return self._original_forward(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

        attn_self.forward = types.MethodType(new_forward, attn_self)




def main(args): 
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    num_train = 128*5
    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    
    dataset = load_dataset("fancyzhx/yelp_polarity")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].select(range(num_train))

    train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

    # Load models
    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    cos = {
        'Activation': [], 'Weight': [],
        'Q': [], 'K': [], 'V': [],
        'V2_Activation': [], 'V2_Weight': [],
        'V2_Q': [], 'V2_K': [], 'V2_V': []
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for reuse_pct in [0.0, 0.5, 0.75, 0.9, 0.99]:
        print(f"=== Running reuse_pct={reuse_pct:.2f} ===")
        # Reset logs
        for key in cosine_logs.keys():
            cosine_logs[key] = 0.0 if isinstance(cosine_logs[key], float) else 0
        cosine_logs['lin_count'] = 0
        cosine_logs['att_count'] = 0

        # Build respropified model with reuse_pct
        model = respropify_bert_att_k(
            base_model,
            att_reuse_schedule=[(reuse_pct, 0)],
            lin_reuse_schedule=[(reuse_pct, 0)],
            lin_k=1,
            att_k=1, 
            omitted_layers={}
        )
        patch_bert_self_attention_k(model)
        model.to(device)
        model.eval()

        # Stock AdamW (unused updates, only backward)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        #Run a few forward/backward passes
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

        # for batch in dataloader:
        #     batch["labels"] = batch.pop("label")
        #     batch = {k: v.to(device) for k, v in batch.items()} 
        #     outputs = model(**batch)
        #     loss = outputs.loss
        #     logits = outputs.logits.detach().cpu()
        #     loss.backward()

        # Aggregate cosine metrics
        cos['Activation'].append(cosine_logs['lin_grad_inputs'] / max(1, cosine_logs['lin_count']))
        cos['Weight'].append(cosine_logs['lin_grad_weights'] / max(1, cosine_logs['lin_count']))
        cos['Q'].append(cosine_logs['att_grad_Q'] / max(1, cosine_logs['att_count']))
        cos['K'].append(cosine_logs['att_grad_K'] / max(1, cosine_logs['att_count']))
        cos['V'].append(cosine_logs['att_grad_V'] / max(1, cosine_logs['att_count']))

        cos['V2_Activation'].append(cosine_logs['lin_v2_inputs'] / max(1, cosine_logs['lin_count']))
        cos['V2_Weight'].append(cosine_logs['lin_v2_weights'] / max(1, cosine_logs['lin_count']))
        cos['V2_Q'].append(cosine_logs['att_v2_Q'] / max(1, cosine_logs['att_count']))
        cos['V2_K'].append(cosine_logs['att_v2_K'] / max(1, cosine_logs['att_count']))
        cos['V2_V'].append(cosine_logs['att_v2_V'] / max(1, cosine_logs['att_count']))

        print_cosine_logs()
        torch.cuda.empty_cache()

    print("\n=== Final Cosine Summary ===")
    for k, v in cos.items():
        print(f"{k}: {v}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./training_data")
    parser.add_argument("--token_path", type=str, default="bert-base-uncased")
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    args = parser.parse_args()
    main(args)