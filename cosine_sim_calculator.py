import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import types
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

def compare_parameters(param1, param2, name=""):
    cos_sim = F.cosine_similarity(param1.view(-1), param2.view(-1), dim=0).item()
    max_diff = (param1 - param2).abs().max().item()
    print(f"{name}:")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  Max absolute difference: {max_diff:.6e}")
def cos_sim(a, b):
    return F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

def cos_sim_batch(a, b, eps=1e-8):
    # a, b: [B, *, *] → Flatten last two dims
    B = a.size(0)
    a_flat = a.view(B, -1)              # [B, T*T]
    b_flat = b.view(1, -1).expand(B, -1)  # [B, T*T], broadcast b

    a_norm = a_flat / (a_flat.norm(dim=-1, keepdim=True) + eps)
    b_norm = b_flat / (b_flat.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1).mean().item()  # [B]

cosines = {}
cosines['lin_grad_weights'] = 0
cosines['lin_grad_inputs'] = 0
cosines['lin_count'] = 0
cosines['att_grad_V'] = 0
cosines['att_grad_K'] = 0
cosines['att_grad_Q'] = 0
cosines['att_count'] = 0
cosines['att_grad_probs'] = 0
cosines['prev_V_prod'] = 0
cosines['prev_attn_prod'] = 0
cosines['prev_K_prod'] = 0
cosines['prev_Q_prod'] = 0
cosines['V_assump'] = 0
cosines['att_probs_assump'] = 0
cosines['V_assump_to_original'] = 0
cosines['att_probs_assump_to_original'] = 0



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
        prev_grad_output = prev_grad_output if prev_grad_output is not None else None

        if prev_grad_output is not None and len(input.shape) == 3 and prev_grad_output.size(0) == input.size(1):
            # Precompute prev_grad_input and prev_grad_weight
            prev_grad_input = torch.mm(prev_grad_output, weight)
            prev_grad_weight = torch.mm(prev_grad_output.t(), torch.sum(input, dim=0))
        else:
            if prev_grad_output is not None:
                print("Warning: Couldn't reuse gradient due to shape mis-match.")
            prev_grad_output = prev_grad_input = prev_grad_weight = None

        ctx.reuse_percentage = reuse_percentage
        ctx.structured = structured
        ctx.n = n
        ctx.group_size = group_size
        ctx.save_for_backward(input, weight, bias, prev_grad_output, prev_grad_input, prev_grad_weight)

        output = torch.bmm(input, weight.t().expand(input.size(0), -1, -1))
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, prev_grad_output, prev_grad_input, prev_grad_weight = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        og_grad_output= grad_output
        # Compute reuse mask
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

        if prev_grad_output is not None:
            og_grad_input = torch.bmm(og_grad_output, weight.expand(og_grad_output.size(0), -1, -1))
            og_grad_weight = torch.sum(torch.bmm(og_grad_output.transpose(1, 2), input), dim=0)
            cosines['lin_grad_inputs']+=  cos_sim(og_grad_input, grad_input)
            cosines['lin_grad_weights']+=  cos_sim(og_grad_weight, grad_weight)
            cosines['lin_count'] += 1

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 1))

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
            reuse_percentage = None
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
                if self.step_counter[device] % self.k == 0:
                    if self.avg:
                        self.prev_gradients[device] = torch.mean(grad_output, dim=0)
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
        ctx.save_for_backward(Q, K, V, prev_grad_output, prev_attn_prod, prev_V_prod, prev_soft_grad, prev_K_prod, prev_Q_prod)

        d_k = Q.size(-1)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        ctx.attn_probs = attn_probs  # current softmax

        return torch.bmm(attn_probs, V)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, prev_grad_output, prev_attn_prod, prev_V_prod, prev_soft_grad, prev_K_prod, prev_Q_prod = ctx.saved_tensors
        
        attn_probs = ctx.attn_probs
        d_k = Q.size(-1)
        grad_Q=grad_K=grad_V=None

        if prev_grad_output is not None and ctx.reuse_percentage > 0:
            grad_diff = generate_reuse_mask(ctx.reuse_percentage, grad_output, prev_grad_output, ctx.structured, ctx.n, ctx.group_size)
            grad_attn_probs = torch.bmm(grad_diff, V.transpose(1, 2)) + prev_V_prod
            if ctx.needs_input_grad[2]:
                grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_diff) + prev_attn_prod


            prev_grad_exp = prev_grad_output.expand(V.size(0), -1, -1)  # [B, T, d_v]
            og_prev_V_prod = torch.bmm(prev_grad_exp, V.transpose(1, 2))
            og_whole_attn_prob = torch.bmm(grad_diff, V.transpose(1, 2)) + og_prev_V_prod
            cosines['att_probs_assump']+=  cos_sim(og_whole_attn_prob, grad_attn_probs)

            og_prev_attn_prod = torch.bmm(attn_probs.transpose(1, 2), prev_grad_exp)
            og_whole_grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_diff) + og_prev_attn_prod
            cosines['V_assump']+=  cos_sim(og_whole_grad_V, grad_V)

        else: 
            grad_attn_probs = torch.bmm(grad_output, V.transpose(1, 2))
            if ctx.needs_input_grad[2]:
                grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_output)

       

        if prev_grad_output is not None:
            og_grad_attn_probs = torch.bmm(grad_output, V.transpose(1, 2))
            og_grad_V = torch.bmm(attn_probs.transpose(1, 2), grad_output)
            cosines['att_grad_probs']+=  cos_sim(og_grad_attn_probs, grad_attn_probs)
            cosines['att_grad_V']+=  cos_sim(og_grad_V, grad_V)
            cosines['att_count'] += 1
            og_grad_attn_scores = og_grad_attn_probs * attn_probs
            og_grad_attn_scores -= attn_probs * og_grad_attn_scores.sum(dim=-1, keepdim=True)
            og_grad_attn_scores /= (d_k ** 0.5)

            if ctx.reuse_percentage > 0:
                cosines['V_assump_to_original'] += cos_sim(og_grad_V, og_whole_grad_V)
                cosines['att_probs_assump_to_original'] += cos_sim(og_grad_attn_probs, og_whole_attn_prob)

            
            # THESE SHOULD BE INDEPENDENT OF THE REUSE_PERCENTAGE
            prev_grad_exp = prev_grad_output.expand(V.size(0), -1, -1)  # [B, T, d_v]
            og_prev_V_prod = torch.bmm(prev_grad_exp, V.transpose(1, 2))
            cosines['prev_V_prod']+=  cos_sim_batch(og_prev_V_prod, prev_V_prod)
            og_prev_attn_prod = torch.bmm(attn_probs.transpose(1, 2), prev_grad_exp)
            cosines['prev_attn_prod']+=  cos_sim_batch(og_prev_attn_prod, prev_attn_prod)


        grad_attn_scores = grad_attn_probs * attn_probs
        grad_attn_scores -= attn_probs * grad_attn_scores.sum(dim=-1, keepdim=True)
        grad_attn_scores /= (d_k ** 0.5)

        if prev_soft_grad is not None and ctx.reuse_percentage > 0:
            grad_attn_diff = generate_reuse_mask(ctx.reuse_percentage, grad_attn_scores, prev_soft_grad, ctx.structured, ctx.n, ctx.group_size)
            if ctx.needs_input_grad[0]:
                grad_Q = torch.bmm(grad_attn_diff, K) + prev_K_prod
            if ctx.needs_input_grad[1]:
                grad_K = torch.bmm(grad_attn_diff.transpose(1, 2), Q) + prev_Q_prod
        else: 
            if ctx.needs_input_grad[0]:
                grad_Q = torch.bmm(grad_attn_scores, K)
            if ctx.needs_input_grad[1]:
                grad_K = torch.bmm(grad_attn_scores.transpose(1, 2), Q)

        if prev_grad_output is not None:
            og_grad_Q = torch.bmm(og_grad_attn_scores, K)
            og_grad_K = torch.bmm(og_grad_attn_scores.transpose(1, 2), Q)
            cosines['att_grad_Q']+=  cos_sim(og_grad_Q, grad_Q)
            cosines['att_grad_K']+=  cos_sim(og_grad_K, grad_K)

            # THESE SHOULD BE INDEPENDENT OF THE REUSE_PERCENTAGE
            prev_soft_grad_exp = prev_soft_grad.expand(K.size(0), -1, -1)
            og_prev_K_prod = torch.bmm(prev_soft_grad_exp, K)
            cosines['prev_K_prod']+=  cos_sim_batch(og_prev_K_prod, prev_K_prod)
            og_prev_Q_prod = torch.bmm(prev_soft_grad_exp.transpose(1, 2), Q)
            cosines['prev_Q_prod']+=  cos_sim_batch(og_prev_Q_prod, prev_Q_prod)

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
            reuse_percentage = None
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
                if self.step_counter[device] % self.k == 0:
                    with torch.no_grad():
                        sampled_grad = grad_output.mean(dim=0, keepdim=True).detach()  # [1, T, d]

                        Q_ = Q.mean(dim=0, keepdim=True)  # [1, T, d_k]
                        K_ = K.mean(dim=0, keepdim=True)  # [1, T, d_k]
                        V_ = V.mean(dim=0, keepdim=True)  # [1, T, d_v]
                        d_k = Q.size(-1)

                        attn_scores = torch.bmm(Q_, K_.transpose(1, 2)) / (d_k ** 0.5)  # [1, T, T]
                        attn_probs = torch.softmax(attn_scores, dim=-1)                 # [1, T, T]

                        attn_prod = torch.bmm(attn_probs.transpose(1, 2), sampled_grad)  # [1, T, d]
                        V_prod = torch.bmm(sampled_grad, V_.transpose(1, 2))             # [1, d, T]

                        grad_softmax = torch.bmm(sampled_grad, V_.transpose(1, 2))       # [1, T, T]
                        grad_attn_scores = grad_softmax * attn_probs                     # [1, T, T]
                        grad_attn_scores -= attn_probs * grad_attn_scores.sum(dim=-1, keepdim=True)
                        grad_attn_scores /= (d_k ** 0.5)

                        K_prod = torch.bmm(grad_attn_scores, K_)                         # [1, T, d]
                        Q_prod = torch.bmm(grad_attn_scores.transpose(1, 2), Q_)         # [1, T, d]

                        self.prev_grad_output[device] = sampled_grad
                        self.prev_attn_prods[device] = attn_prod.detach()
                        self.prev_V_prods[device] = V_prod.detach()
                        self.prev_soft_grads[device] = grad_attn_scores.detach()
                        self.prev_K_prods[device] = K_prod.detach()
                        self.prev_Q_prods[device] = Q_prod.detach()

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


def print_cosine_stats(): 
    print(f"cosine similarity linear weight: {cosines['lin_grad_weights']/cosines['lin_count']}")
    print(f"cosine similarity linear input: {cosines['lin_grad_inputs']/cosines['lin_count']}")
    print(f"cosine similarity attention V: {cosines['att_grad_V']/cosines['att_count']}")
    print(f"cosine similarity attention K: {cosines['att_grad_K']/cosines['att_count']}")
    print(f"cosine similarity attention Q: {cosines['att_grad_Q']/cosines['att_count']}")
    print(f"cosine similarity attention input: {cosines['att_grad_probs']/cosines['att_count']}")
    print(f"cosine similarity attention precompute attn prod: {cosines['prev_attn_prod']/cosines['att_count']}")
    print(f"cosine similarity attention precompute V prod: {cosines['prev_V_prod']/cosines['att_count']}")
    print(f"cosine similarity attention precompute K prod: {cosines['prev_K_prod']/cosines['att_count']}")
    print(f"cosine similarity attention precompute Q prod: {cosines['prev_Q_prod']/cosines['att_count']}")
    print(f"cosine similarity attention V assumption: {cosines['V_assump']/cosines['att_count']}")
    print(f"cosine similarity attention probs assumption: {cosines['att_probs_assump']/cosines['att_count']}")
    print(f"cosine similarity attention V assumption to original: {cosines['V_assump_to_original']/cosines['att_count']}")
    print(f"cosine similarity attention probs assumption to original: {cosines['att_probs_assump_to_original']/cosines['att_count']}")


def main(): 
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("fancyzhx/yelp_polarity")

    num_train = 128*5
    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].select(range(num_train))

    train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

    # Load models
    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="eager",
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )
    cos = {}
    cos['Activation'] = []
    cos['Weight'] = []
    cos['Q'] = []
    cos['K'] = []
    cos['V'] = []
    for reuse_pct in [0, 0.5, 0.75, 0.9, 0.99]: 
        for key in cosines.keys():
            cosines[key] = 0
        model = respropify_bert_att_k(base_model, att_reuse_schedule=[(reuse_pct, 0)], lin_reuse_schedule=[(reuse_pct, 0)], lin_k=1, att_k=1)
        patch_bert_self_attention_k(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        for batch in dataloader:
            batch["labels"] = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()} 
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits.detach().cpu()
            loss.backward()
        
        cos['Activation'].append(cosines['lin_grad_inputs']/cosines['lin_count'])
        cos['Weight'].append(cosines['lin_grad_weights']/cosines['lin_count'])
        cos['Q'].append(cosines['att_grad_Q']/cosines['att_count'])
        cos['K'].append(cosines['att_grad_K']/cosines['att_count'])
        cos['V'].append(cosines['att_grad_V']/cosines['att_count'])
        # print_cosine_stats()
    print(cos)


if __name__ == "__main__": 
    main()