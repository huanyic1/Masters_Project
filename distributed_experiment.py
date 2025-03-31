import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from resprop_linear import resprofify_bert
from plot import plot_log_histories
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Or any free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process(rank, world_size, reuse_percentage, model_name, train_dataset, eval_dataset):
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}, Reuse Percentage: {reuse_percentage}")

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    model = resprofify_bert(base_model, reuse_percentage=reuse_percentage)
    model.to(device)
    # model = DDP(model, device_ids=[rank]) #wrap with DDP

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}/rank_{rank}",
        eval_strategy="steps",
        num_train_epochs=5,
        eval_steps=1/16,
        logging_steps=1/16,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}/rank_{rank}",
        per_device_train_batch_size=8, #adjust batch size to fit on each GPU
        per_device_eval_batch_size=8,
        # local_rank=rank, #required for DDP
    )

    metric = evaluate.load("accuracy")

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
        torch.save(trainer.state.log_history, f"log_history_{reuse_percentage}.pt")
    # cleanup()

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    model_name = "prajjwal1/bert-tiny"
    # model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("fancyzhx/yelp_polarity")

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(32000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    reuse_percentages = [0.0, 0.99]
    world_size = torch.cuda.device_count()

    for reuse_percentage in reuse_percentages:
        mp.spawn(
            process,
            args=(world_size, reuse_percentage, model_name, train_dataset, eval_dataset),
            nprocs=world_size
        )

    log_histories = {}
    for reuse_percentage in reuse_percentages:
        log_histories[reuse_percentage] = torch.load(f"log_history_{reuse_percentage}.pt")

    plot_log_histories(log_histories, file_name="result_ddp.png")