# train_resprop.py

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from resprop_attention_k import respropify_bert_att_k, patch_bert_self_attention_k
from evaluate import load as load_metric
import os
import torch.distributed as dist
import json

reuse_schedule_path = "reuse_schedule.json"
REUSE_SCHEDULES = json.load(open(reuse_schedule_path))
output_dir = "outputs"


def cleanup():
    dist.destroy_process_group()

def get_schedule_string(schedule):
    return ','.join(f'(rp: {x}, start: {y})' for x, y in schedule)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse_schedule_idx', type=int, required=True)
    parser.add_argument('--att_k', type=int, default=1)
    parser.add_argument('--lin_k', type=int, default=1)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--model_name', type=str, default="prajjwal1/bert-tiny")
    parser.add_argument('--num_train', type=int, default=3200)
    parser.add_argument('--num_test', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5)

    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.cuda.device_count()

    device = torch.device(f"cuda:{rank}")
    model_name = args.model_name

    att_schedule = REUSE_SCHEDULES[0][args.reuse_schedule_idx]
    lin_schedule = REUSE_SCHEDULES[1][args.reuse_schedule_idx]
    att_str = get_schedule_string(att_schedule)
    lin_str = get_schedule_string(lin_schedule)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("fancyzhx/yelp_polarity")

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=args.seed).select(range(args.num_train))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=args.seed).select(range(args.num_test))

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_iters = (len(train_dataset) // world_size) * num_epochs // batch_size
    eval_steps = max(1, num_iters // 16)

    scaled_lin_schedule = [(x, y * len(train_dataset) // world_size) for x, y in lin_schedule]
    scaled_att_schedule = [(x, y * len(train_dataset) // world_size) for x, y in att_schedule]

    model = respropify_bert_att_k(base_model, att_reuse_schedule=scaled_att_schedule, lin_reuse_schedule=scaled_lin_schedule, lin_k=args.lin_k, att_k=args.att_k)
    patch_bert_self_attention_k(model)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_att_{att_str}_lin_{lin_str}_seed_{args.seed}_att_k_{args.att_k}_lin_k_{args.lin_k}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        seed=args.seed,

    )

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    if rank == 0:
        torch.save(trainer.state.log_history,
                   f"{output_dir}/log_history_att_{att_str}_lin_{lin_str}_seed_{args.seed}_att_k_{args.att_k}_lin_k_{args.lin_k}.pt")


    cleanup()

if __name__ == "__main__":
    main()