#!/usr/bin/env python3
import os
import argparse

import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    BertConfig,
    BertForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def parse_args():
    p = argparse.ArgumentParser(description="Resume BERT pre-training with torchrun + DDP")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to your semi-trained BERT (folder with config & pytorch_model.bin)",
    )
    p.add_argument(
        "--token_path",
        type=str,
        default='bert-base-uncased',
        help="Path to your semi-trained BERT (folder with config & pytorch_model.bin)",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="cached_datasets",
        help="Where you saved wiki-... and openwebtext-... via `Dataset.save_to_disk`",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="resume_outputs",
        help="Where to write new checkpoints and final model",
    )
    p.add_argument(
        "--batch_size", type=int, default=32, help="Per-GPU batch size"
    )
    p.add_argument(
        "--epochs", type=int, default=3, help="Total epochs to train"
    )
    p.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Initial learning rate"
    )
    p.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Masking probability for MLM",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of workers for DataLoader / map",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load cached datasets & concatenate
    wiki = load_from_disk(os.path.join(args.cache_dir, "wikipedia-20220301.en-train"))
    owt  = load_from_disk(os.path.join(args.cache_dir, "openwebtext-None-train"))
    train_ds = concatenate_datasets([wiki, owt])

    # 2) tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=True)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=128,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = BertForMaskedLM(config)

    # 3) data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # 4) TrainingArguments picks up LOCAL_RANK from torchrun automatically
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        save_total_limit=3,
        save_steps=10_000,
        logging_steps=500,
        dataloader_num_workers=args.num_proc,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        local_rank=local_rank,
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    # 6) resume training
    trainer.train()

    # 7) final save
    trainer.save_model(args.output_dir)
    if local_rank in (-1, 0):
        print(f"✅ Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

# import argparse
# import logging
# import os
# import torch
# from accelerate import Accelerator
# from datasets import load_from_disk, concatenate_datasets
# from transformers import (
#     BertConfig,
#     BertForPreTraining,
#     AutoTokenizer,
#     DataCollatorForLanguageModeling,
#     get_scheduler,
#     set_seed,
# )
# from torch.optim import AdamW
# from tqdm.auto import tqdm

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output_dir", type=str, default="./bert_from_scratch")
#     parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
#     parser.add_argument("--dataset_cache_dir", type=str, default="./cached_openweb")
#     parser.add_argument("--block_size", type=int, default=128)
#     parser.add_argument("--tokenizer_name_or_path", type=str, default="bert-base-uncased")
#     parser.add_argument("--mlm_probability", type=float, default=0.15)
#     parser.add_argument("--per_device_train_batch_size", type=int, default=128)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
#     parser.add_argument("--learning_rate", type=float, default=5e-5)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--steps_per_shard", type=int, default=10000)
#     parser.add_argument("--lr_scheduler_type", type=str, default="linear",
#                         choices=["linear", "cosine", "polynomial", "constant", "constant_with_warmup"])
#     parser.add_argument("--num_warmup_steps", type=int, default=0)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--with_tracking", action="store_true", default=True)
#     parser.add_argument("--report_to", type=str, default="all")
#     parser.add_argument("--resume_from_checkpoint", action="store_true")
#     parser.add_argument("--reset_shard_id", action="store_true")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     accelerator = Accelerator(log_with=args.report_to, project_dir=args.output_dir)

#     if accelerator.is_main_process:
#         logger.setLevel(logging.INFO)
#     else:
#         logger.setLevel(logging.ERROR)
#     set_seed(args.seed)

#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(args.checkpoint_dir, exist_ok=True)

#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

#     logger.info("Loading preprocessed dataset...")
#     owt = load_from_disk(os.path.join(args.dataset_cache_dir, "owt_preprocessed"))
#     wiki = load_from_disk(os.path.join(args.dataset_cache_dir, "wiki_preprocessed"))
#     dataset = concatenate_datasets([owt, wiki])
#     print('DATASET SHAPE', dataset.shape)

#     logger.info("Filtering sequences of incorrect length...")
#     # Filter sequences where input_ids isn't exactly block_size
#     expected_length = args.block_size
#     def is_valid(example):
#         return (
#             len(example["input_ids"]) == args.block_size and
#             len(example.get("attention_mask", [])) == args.block_size and
#             len(example.get("token_type_ids", [])) == args.block_size
#         )

#     dataset = dataset.filter(is_valid, num_proc=4)

#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=True,
#         mlm_probability=0.15,
#         pad_to_multiple_of=args.block_size,  # Optional: aligns for tensor cores
#     )
#     config = BertConfig(
#         vocab_size=tokenizer.vocab_size,
#         hidden_size=768,
#         num_hidden_layers=12,
#         num_attention_heads=12,
#         intermediate_size=3072,
#         max_position_embeddings=args.block_size,
#         type_vocab_size=2,
#         pad_token_id=tokenizer.pad_token_id,
#     )

#     model = BertForPreTraining(config)

#     if args.with_tracking:
#         accelerator.init_trackers("bert_pretraining_from_scratch")

#     train_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.per_device_train_batch_size,
#         collate_fn=data_collator,
#         shuffle=True,
#         num_workers=2,
#     )

#     optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     lr_scheduler = get_scheduler(
#         name=args.lr_scheduler_type,
#         optimizer=optimizer,
#         num_warmup_steps=args.num_warmup_steps,
#         num_training_steps=args.steps_per_shard,
#     )

#     model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
#     model.train()

#     completed_steps = 0
#     progress_bar = tqdm(range(args.steps_per_shard), disable=not accelerator.is_main_process)

#     for step, batch in enumerate(train_loader):
#         if completed_steps >= args.steps_per_shard:
#             break

#         batch = {k: v.to(accelerator.device) for k, v in batch.items()}
#         batch.setdefault("token_type_ids", torch.zeros_like(batch["input_ids"]))
#         batch.setdefault("next_sentence_label", torch.zeros(batch["input_ids"].shape[0], dtype=torch.long))

#         allowed_keys = {"input_ids", "attention_mask", "token_type_ids", "next_sentence_label", "labels"}
#         filtered_batch = {k: v for k, v in batch.items() if k in allowed_keys}

#         print("input_ids shape:", batch["input_ids"].shape)

#         if batch["input_ids"].dim() == 3 and batch["input_ids"].size(1) == 1:
#             batch["input_ids"] = batch["input_ids"].squeeze(1)

#         loss = model(**filtered_batch).loss / args.gradient_accumulation_steps
#         accelerator.backward(loss)

#         if step % args.gradient_accumulation_steps == 0:
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             progress_bar.update(1)
#             completed_steps += 1

#             if accelerator.is_main_process:
#                 accelerator.log({"loss": loss.item() * args.gradient_accumulation_steps}, step=completed_steps)

#     accelerator.wait_for_everyone()
#     if accelerator.is_main_process:
#         unwrapped_model = accelerator.unwrap_model(model)
#         save_path = os.path.join(args.checkpoint_dir, "checkpoint_final")
#         unwrapped_model.save_pretrained(save_path, save_function=accelerator.save)
#         tokenizer.save_pretrained(save_path)
#         logger.info(f"Saved final checkpoint at {save_path}")
#         if args.with_tracking:
#             accelerator.end_training()

# if __name__ == "__main__":
#     main()