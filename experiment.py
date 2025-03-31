import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import copy
from transformers import Trainer, TrainingArguments, AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset
from resprop_linear import resprofify_bert
import evaluate

def setup(rank, world_size):
    """Initialize DistributedDataParallel (DDP)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup(rank):
    rank = dist.get_rank()
    
    """Cleanup after training."""
    dist.barrier(device_ids=[rank])  # ✅ Ensure all processes sync before exiting
    dist.destroy_process_group()
def train(rank, world_size):
    setup(rank, world_size)

    model_name = "prajjwal1/bert-tiny"
    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    ).to(rank)
    if hasattr(base_model, "classifier"):
        base_model.classifier.reset_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    dataset = load_dataset("fancyzhx/yelp_polarity")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(4000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    ### **🔥 Assign Different `reuse_percentage` Based on `rank`** 🔥 ###
    reuse_percentages = [0.0, 0.9, 0.99]  # Customize per GPU
    reuse_percentage = reuse_percentages[rank % len(reuse_percentages)]  # Cycle through values

    print(f"Rank {rank}: Using reuse_percentage = {reuse_percentage}")

    # Synchronize resprofify_bert across all GPUs
    model = resprofify_bert(base_model, reuse_percentage=reuse_percentage).to(rank)
    if dist.is_initialized():
        dist.barrier(device_ids=[rank])  # Ensure GPUs synchronize properly # Ensure all GPUs have the same model

    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}",
        eval_strategy="steps",
        num_train_epochs=4,
        eval_steps=1/64,
        logging_steps=1/32,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}",
        local_rank=rank,  # ✅ Ensure DDP is correctly detected
        ddp_find_unused_parameters=False,  # ✅ Optimize DDP usage
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of available GPUs
    train(0, world_size)