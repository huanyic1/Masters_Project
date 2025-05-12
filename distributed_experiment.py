import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from resprop_linear import resprofify_bert, resprofify_bert_warmup
from resprop_attention import respropify_bert_att, patch_bert_self_attention
from resprop_attention_k import respropify_bert_att_k, patch_bert_self_attention_k
from resprop_linear_k import resprofify_k_bert
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

def process_base(rank, world_size, reuse_percentage, model_name, train_dataset, eval_dataset):
    # FOR ABSOLUTELY NO CHANGES
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}, Reuse Percentage: {reuse_percentage}")

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )
    model = base_model
    model.to(device)
    # model = DDP(model, device_ids=[rank]) #wrap with DDP

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/baseline",
        eval_strategy="steps",
        num_train_epochs=10,
        eval_steps=1/16,
        logging_steps=1/16,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/baseline",
        per_device_train_batch_size=8, #adjust batch size to fit on each GPU
        per_device_eval_batch_size=8,
        seed=42,
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
        torch.save(trainer.state.log_history, f"log_history_baseline.pt")

def process_k(rank, world_size, reuse_percentage, model_name, train_dataset, eval_dataset, k):
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}, Reuse Percentage: {reuse_percentage}")

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    model = resprofify_k_bert(base_model, reuse_percentage=reuse_percentage, k=k)
    model.to(device)
    # model = DDP(model, device_ids=[rank]) #wrap with DDP

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}_{k}/rank_{rank}",
        eval_strategy="steps",
        num_train_epochs=5,
        eval_steps=1/16,
        logging_steps=1/16,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}_{k}/rank_{rank}",
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
        torch.save(trainer.state.log_history, f"log_history_{reuse_percentage}_{k}.pt")
    # cleanup()

def process(rank, world_size, reuse_percentage, model_name, train_dataset, eval_dataset, seed):
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
        seed=seed,
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

def process_warmup(rank, world_size, reuse_schedule, model_name, train_dataset, eval_dataset):
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}")

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    num_samples = len(train_dataset)
    num_epochs = 5
    num_samples_per_rank = num_samples // world_size
    batch_size = 8
    num_iters = num_samples_per_rank * num_epochs // batch_size
    scaled_reuse_schedule = [(x * num_samples_per_rank, y) for x, y in reuse_schedule]
    print('NUMBER OF ITERATIONS', num_iters)

    model = resprofify_bert_warmup(base_model, reuse_schedule=scaled_reuse_schedule)
    model.to(device)
    # model = DDP(model, device_ids=[rank]) #wrap with DDP

   

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/warmup/rank_{rank}",
        eval_strategy="steps",
        num_train_epochs=num_epochs,
        eval_steps=1/16,
        logging_steps=1/16,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/warmup/rank_{rank}",
        per_device_train_batch_size=batch_size, #adjust batch size to fit on each GPU
        per_device_eval_batch_size=batch_size,
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
        torch.save(trainer.state.log_history, f"log_history_warmup.pt")

def att_process(rank, world_size, lin_reuse_percentage, att_reuse_percentage, model_name, train_dataset, eval_dataset, seed=42, lin_k=1, att_k=1):
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}, Attention Reuse Percentage: {att_reuse_percentage}, Linear Reuse Percentage: {lin_reuse_percentage}")

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    # model = resprofify_bert(base_model, reuse_percentage=lin_reuse_percentage)
    model = respropify_bert_att_k(base_model, att_reuse_percentage=att_reuse_percentage, lin_reuse_percentage=lin_reuse_percentage, lin_k=lin_k, att_k=att_k)
    patch_bert_self_attention_k(model)
    model.to(device)
    # model = DDP(model, device_ids=[rank]) #wrap with DDP

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_att_{int(att_reuse_percentage * 100)}_lin_{int(lin_reuse_percentage * 100)}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}/rank_{rank}", 
        eval_strategy="steps",
        num_train_epochs=5,
        eval_steps=1/16,
        logging_steps=1/16,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_att_{int(att_reuse_percentage * 100)}_lin_{int(lin_reuse_percentage * 100)}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}/rank_{rank}",
        per_device_train_batch_size=8, #adjust batch size to fit on each GPU
        per_device_eval_batch_size=8,
        seed=seed,
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
        torch.save(trainer.state.log_history, f"log_history_att_{att_reuse_percentage}_lin_{lin_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}.pt")
    # cleanup()

def att_process_new(rank, world_size, lin_reuse_percentage, att_reuse_percentage, model_name, train_dataset, eval_dataset, seed=42, lin_k=1, att_k=1):
    # setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}, Attention Reuse Percentage: {att_reuse_percentage}, Linear Reuse Percentage: {lin_reuse_percentage}")

    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"}
    )

    # model = resprofify_bert(base_model, reuse_percentage=lin_reuse_percentage)
    model = respropify_bert_att(base_model, att_reuse_percentage=att_reuse_percentage, lin_reuse_percentage=lin_reuse_percentage, lin_k=lin_k, att_k=att_k)
    patch_bert_self_attention(model)
    model.to(device)
    # model = DDP(model, device_ids=[rank]) #wrap with DDP

    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_att_{int(att_reuse_percentage * 100)}_lin_{int(lin_reuse_percentage * 100)}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}/rank_{rank}_new", 
        eval_strategy="steps",
        num_train_epochs=5,
        eval_steps=1/16,
        logging_steps=1/16,
        logging_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_att_{int(att_reuse_percentage * 100)}_lin_{int(lin_reuse_percentage * 100)}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}/rank_{rank}_new",
        per_device_train_batch_size=8, #adjust batch size to fit on each GPU
        per_device_eval_batch_size=8,
        seed=seed,
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
        torch.save(trainer.state.log_history, f"new_log_history_att_{att_reuse_percentage}_lin_{lin_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}.pt")
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


    seed = 45
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shuffle(seed=seed).select(range(3200))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=seed).select(range(1000))

    lin_reuse_percentages = [0.9]
    att_reuse_percentages = [0.9]
    att_k_s = [1,2, 3, 4, 5]
    lin_k_s = [1,2,3,4, 5, 10]
    world_size = torch.cuda.device_count()


    # reuse_schedule = [(0, 0.99), (0.1, 0.90)]
    # mp.spawn(
    #     process_warmup,
    #     args=(world_size, reuse_schedule, model_name, train_dataset, eval_dataset),
    #     nprocs=world_size
    # )
    
    log_histories = {}
    # lin_reuse_percentage = 0
    # att_reuse_percentage = 0.9
    # lin_k = 1
    # att_k = 2

    # mp.spawn(
    #     att_process,
    #     args=(world_size, lin_reuse_percentage, att_reuse_percentage, model_name, train_dataset, eval_dataset, seed, lin_k, att_k),
    #     nprocs=world_size
    # )
    # log_histories[f"lin_{lin_reuse_percentage}_att_{att_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}"] = torch.load(f"log_history_att_{att_reuse_percentage}_lin_{lin_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}.pt")


    
    for reuse_percentage in lin_reuse_percentages:
        for k in lin_k_s:
            lin_reuse_percentage = 0
            att_reuse_percentage = reuse_percentage
            lin_k = 1
            att_k = k

            mp.spawn(
                att_process,
                args=(world_size, lin_reuse_percentage, att_reuse_percentage, model_name, train_dataset, eval_dataset, seed, lin_k, att_k),
                nprocs=world_size
            )
            log_histories[f"lin_{lin_reuse_percentage}_att_{att_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}"] = torch.load(f"log_history_att_{att_reuse_percentage}_lin_{lin_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}.pt")

            # mp.spawn(
            #     att_process_new,
            #     args=(world_size, lin_reuse_percentage, att_reuse_percentage, model_name, train_dataset, eval_dataset, seed, lin_k, att_k),
            #     nprocs=world_size
            # )
            # log_histories[f"new_lin_{lin_reuse_percentage}_att_{att_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}"] = torch.load(f"new_log_history_att_{att_reuse_percentage}_lin_{lin_reuse_percentage}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}.pt")

            # mp.spawn(
            #     process_k,
            #     args=(world_size, reuse_percentage, model_name, train_dataset, eval_dataset, k),
            #     nprocs=world_size
            # )
            # log_histories[f"{reuse_percentage}_linear_k_{k}"] = torch.load(f"log_history_{reuse_percentage}_{k}.pt")



    plot_log_histories(log_histories, file_name=f"resprop_k_both.png")
    # k_reuse_percentages = [0.5, 0.7, 0.9]
    # k_s = [1, 2, 3, 4, 5]


    # for reuse_percentage in k_reuse_percentages:
    #     log_histories = {}
    #     for k in k_s:
    #         mp.spawn(
    #             process_k,
    #             args=(world_size, reuse_percentage, model_name, train_dataset, eval_dataset, k),
    #             nprocs=world_size
    #         )
    #         log_histories[(reuse_percentage, k)] = torch.load(f"log_history_{reuse_percentage}_{k}.pt")
    #     plot_log_histories(log_histories, file_name=f"result_ddp_rp_{int(reuse_percentage * 100)}.png")

    # for reuse_percentage in reuse_percentages:
    #     log_histories[(reuse_percentage, 0)] = torch.load(f"log_history_{reuse_percentage}.pt")

    # for reuse_percentage in k_reuse_percentages:
    #     for k in k_s:
            

    # plot_log_histories(log_histories, file_name="result_ddp.png")