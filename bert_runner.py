#!/usr/bin/env python3
import os
import argparse
from resprop_attention_k import respropify_bert_att_k, patch_bert_self_attention_k, ReSpropAttention, ReSpropLinear
from resprop_linear import respropify_bert
from plot_from_trainer import plot_loss
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
import json

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
    p.add_argument('--max_training_samples', type=int, default=None)
    p.add_argument("--resume_path", type=str, default=None)
    p.add_argument("--resume_step", type=int, default=0)
    p.add_argument('--plot_name', type=str, default='trial.png')
    p.add_argument('--baseline',  action='store_true')
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load cached datasets & concatenate
    # wiki = load_from_disk(os.path.join(args.cache_dir, "wikipedia-20220301.en-train"))
    # owt  = load_from_disk(os.path.join(args.cache_dir, "openwebtext-None-train"))
    # train_ds = concatenate_datasets([wiki, owt])
    train_ds_dict =  load_from_disk(os.path.join(args.cache_dir, "wiki_owt_block128"))

    train_ds = train_ds_dict['train']
    if args.max_training_samples is not None:
        max_samples = min(len(train_ds), args.max_training_samples)
        train_ds = train_ds.select(range(max_samples))

    # 2) tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=True)


    att_schedule = [[0.0, 0.0]]
    lin_schedule = [[0.0, 0.0]]
    num_epochs = args.epochs
    batch_size = args.batch_size


    num_iters = (len(train_ds) // args.num_proc) * num_epochs // batch_size

    scaled_lin_schedule = [(x, y * num_iters) for x, y in lin_schedule]
    scaled_att_schedule = [(x, y * num_iters) for x, y in att_schedule]

    if not args.resume_path:
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
        base_model = BertForMaskedLM(config)
        print("generating raw model")
    else:
        base_model = BertForMaskedLM.from_pretrained(args.resume_path)

    if args.baseline:
        model = base_model
        print("Baseline Model")
    else: 
        print("Sharting the Model")
        model = respropify_bert_att_k(base_model, att_reuse_schedule=scaled_att_schedule, lin_reuse_schedule=scaled_lin_schedule, lin_k=1, att_k=1)
        patch_bert_self_attention_k(model)

    # model = respropify_bert(base_model, reuse_schedule=scaled_lin_schedule)
    if args.resume_path:
        # Get step from trainer state
        trainer_state_file = os.path.join(args.resume_path, "trainer_state.json")
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, "r") as f:
                trainer_state = json.load(f)
            resume_step = trainer_state.get("global_step", 0)
        else:
            print("⚠️ Warning: trainer_state.json not found. Defaulting to resume_step = 0")
            resume_step = 0

        # Set step counter in ReSprop modules
        for module in model.modules():
            if isinstance(module, (ReSpropLinear, ReSpropAttention)):
                for device in module.step_counter:
                    module.step_counter[device] = resume_step


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
        fp16=False,
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
    trainer.train(resume_from_checkpoint=args.resume_path)

    # 7) final save
    trainer.save_model(args.output_dir)
    if local_rank in (-1, 0):
        print(f"✅ Training complete. Model saved to {args.output_dir}")

    history = trainer.state.log_history
    plot_loss(history, args.plot_name)

if __name__ == "__main__":
    main()
