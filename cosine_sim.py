import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from resprop_attention_k import respropify_bert_att_k, patch_bert_self_attention_k
from resprop_linear import respropify_bert
import argparse
import os
import copy
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


from torch.autograd import gradcheck


def compare_parameters(param1, param2, name=""):
    cos_sim = F.cosine_similarity(param1.view(-1), param2.view(-1), dim=0).item()
    max_diff = (param1 - param2).abs().max().item()
    print(f"{name}:")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  Max absolute difference: {max_diff:.6e}")

def compare_attention_outputs(baseline_model, custom_model, hidden_states, attention_mask, layer_idx=0):
    device = hidden_states.device

    # === Extract layer modules ===
    # baseline_attn = baseline_model.bert.encoder.layer[layer_idx].attention.self
    # custom_attn   = custom_model.bert.encoder.layer[layer_idx].attention.self
    baseline_attn = custom_model.bert.encoder.layer[layer_idx].attention.self
    custom_attn = custom_model.bert.encoder.layer[layer_idx].attention.self.custom_attention

    # === Run both attention modules ===
    def run_attention(attn_layer):
        return attn_layer.forward(hidden_states)
        return attn_layer(
            hidden_states=hidden_states, 
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False
        )[0]

    baseline_out = run_attention(baseline_attn)
    print("BASELINE", baseline_out[0].shape)
    custom_out   = run_attention(custom_attn)
    print("CUSTOM", custom_out[0].shape)
    # print(baseline_out)

    # === Compare ===
    cos_sim = F.cosine_similarity(baseline_out[0].view(-1), custom_out[0].view(-1), dim=0).item()
    max_diff = (baseline_out[0] - custom_out[0]).abs().max().item()

    print(f"[Layer {layer_idx}] Cosine similarity: {cos_sim:.8f}")
    print(f"[Layer {layer_idx}] Max abs difference: {max_diff:.6e}")

    return baseline_out, custom_out

def cosine_similarity(t1, t2):
    return F.cosine_similarity(t1.view(-1), t2.view(-1), dim=0).item()

def register_grad_hook(model, tag, grads, layers=[0, 11], baseline=False):
    for layer_idx in layers:
        attn = model.bert.encoder.layer[layer_idx].attention.self
        layer = model.bert.encoder.layer[layer_idx]
        if baseline:
            param = attn.query.weight
            #param = layer.output.dense.weight
        else:
            param = attn.custom_attention.q_proj.weight
            #param = layer.output.dense.weight

        def make_hook(name):
            def hook(grad):
                grads[tag][name] = grad.detach().clone()
            return hook
        
        param.register_hook(make_hook(f"layer{layer_idx}"))

def run_batch(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()} 
    outputs = model(**batch)
    loss = outputs.loss
    logits = outputs.logits.detach().cpu()
    loss.backward()
    return loss.item(), logits

def collate_batch(tokenizer, batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")


def generate_dummy_bert_input(model_name="bert-base-uncased", batch_size=128, seq_len=128, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate fake sentences (you can make them more diverse if needed)
    dummy_texts = ["hello world"] * batch_size

    # Tokenize
    encoded = tokenizer(dummy_texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")

    # Dummy labels (binary classification: 0 or 1)
    labels = torch.randint(0, 2, (batch_size,))

    # Move to device
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device)
    }

    return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    parser.add_argument("--num_train", type=int, default=256)
    parser.add_argument("--num_test", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_output", default=False)
    parser.add_argument("--compare_weights", default=False)
    args = parser.parse_args()

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("fancyzhx/yelp_polarity")

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].select(range(args.num_train))
    
    # Convert to torch tensors
    train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

    # Load models
    base_model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )
    custom_model = copy.deepcopy(base_model)
    custom_model = respropify_bert(custom_model, reuse_schedule=[(0.5, 0)])
    #custom_model = respropify_bert_att_k(custom_model, att_reuse_schedule=[(0,0)],lin_reuse_schedule=[(0,0)])
    #patch_bert_self_attention_k(custom_model)
    device = torch.device("cuda")
    custom_model.to(device)
    base_model.to(device)
    base_model.eval()
    custom_model.eval()

    if args.compare_weights:
        for i in range(12):
            base_layer = base_model.bert.encoder.layer[i].attention.self
            custom_layer = custom_model.bert.encoder.layer[i].attention.self
            #compare_parameters(base_layer.value.weight, custom_layer.custom_attention.v_proj.weight)
            compare_parameters(base_model.bert.encoder.layer[i].output.dense.weight, custom_model.bert.encoder.layer[i].output.dense.weight, name=f"Layer {i} - og")

    grads = {
        "baseline": {},
        "custom": {}
    }


    layers = list(range(12))
    register_grad_hook(base_model, "baseline", grads, baseline=True,layers=layers)
    register_grad_hook(custom_model, "custom", grads, baseline=True, layers=layers)

    for i, batch in enumerate(dataloader):
        if i >= 2:
            break

        input_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["label"]
        }

        if i == 0:
            # base_model.zero_grad()
            base_loss, base_logits = run_batch(base_model, input_batch, device)

            # custom_model.zero_grad()
            custom_loss, custom_logits = run_batch(custom_model, input_batch, device)

            if args.compare_output:
                out_sim = cosine_similarity(base_logits, custom_logits)
                print(f"Cosine Similarity of Logits (Batch 0): {out_sim:.6f}")

            print("\nCosine Similarity of Gradients on First Batch:")
            for layer in layers:
                g1 = grads["baseline"][f"layer{layer}"]
                g2 = grads["custom"][f"layer{layer}"]
                sim = cosine_similarity(g1, g2)
                print(f"Layer {layer} (output.dense.weight): {sim:.6f}")



        elif i == 1:
            # base_model.zero_grad()
            base_loss, base_logits = run_batch(base_model, input_batch, device)

            # custom_model.zero_grad()
            custom_loss, custom_logits = run_batch(custom_model, input_batch, device)
            if args.compare_output:
                out_sim = cosine_similarity(base_logits, custom_logits)
                print(f"Cosine Similarity of Logits (Batch 1): {out_sim:.6f}")

            print("\nCosine Similarity of Gradients on Second Batch:")
            for layer in layers:
                g1 = grads["baseline"][f"layer{layer}"]
                g2 = grads["custom"][f"layer{layer}"]
                sim = cosine_similarity(g1, g2)
                print(f"Layer {layer} (output.dense.weight): {sim:.6f}")
    
if __name__ == "__main__":
    main()