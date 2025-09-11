#!/usr/bin/env python3
import argparse
import os
import random
from typing import Dict, List

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, DownloadConfig
from transformers import AutoTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_raw_wikipedia(split: str = "train", wiki_config: str = "20220301.en"):
    # Wikipedia uses a single 'train' split in HF datasets for dumps
    ds = load_dataset("wikipedia", wiki_config, split=split)
    # Some entries can have empty text; filter them out
    ds = ds.filter(lambda x: x.get("text", "").strip() != "")
    # Keep only the 'text' column
    if "text" not in ds.column_names:
        raise ValueError(f"Expected column 'text' in wikipedia dataset; got {ds.column_names}")
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds


def load_raw_openwebtext(split="train"):
    config = DownloadConfig(
        cache_dir="./clean_cache",             # force use of clean location
    )
    ds = load_dataset("text", data_files=os.path.expanduser("~/datasets/openwebtext_extracted/openwebtext/decompressed/*.txt")) # had to do some hacky work around dowloading directly from huggingface and converting
    ds = ds[split]  # Select the 'train' split
    ds = ds.filter(lambda x: x.get("text", "").strip() != "")
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds

def tokenize_function(examples: Dict[str, List[str]], tokenizer, keep_newlines: bool = False):
    texts = examples["text"]
    if not keep_newlines:
        texts = [t.replace("\n", " ") for t in texts]
    return tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,  
        truncation=True,
        padding="max_length",
        max_length=512
    )


def group_texts(examples: Dict[str, List[List[int]]], block_size: int):
    # Concatenate then split into blocks of exactly block_size
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)
    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": []}  # drop too-short batches
    result = {
        "input_ids": [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
    }
    # For plain LM pretraining on BERT-style masked LM, attention_mask is optional;
    # Trainer/DataCollatorForLanguageModeling will create labels via masking.
    return result


def build_and_save(
    out_dir: str,
    tokenizer_name: str = "bert-base-uncased",
    block_size: int = 128,
    num_proc: int = 4,
    wiki_config: str = "20220301.en",
    keep_newlines: bool = False,
    max_wiki_samples: int = None,
    max_owt_samples: int = None,
    seed: int = 42,
):
    set_seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    print(">> Loading tokenizer:", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Ensure block_size fits the tokenizer’s max length
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length and tokenizer.model_max_length > 0:
        model_max = tokenizer.model_max_length
        if block_size > model_max and model_max != int(1e30):  # some tokenizers set a huge sentinel value
            print(f"!! block_size {block_size} > tokenizer.model_max_length {model_max}. Clamping.")
            block_size = model_max

    print(">> Loading raw Wikipedia")
    wiki = load_raw_wikipedia(split="train", wiki_config=wiki_config)
    if max_wiki_samples is not None:
        wiki = wiki.shuffle(seed=seed).select(range(min(max_wiki_samples, len(wiki))))

    print(">> Loading raw OpenWebText")
    owt = load_raw_openwebtext(split="train")
    if max_owt_samples is not None:
        owt = owt.shuffle(seed=seed).select(range(min(max_owt_samples, len(owt))))

    # Tokenize
    print(">> Tokenizing (batched)")
    tokenized_wiki = wiki.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "keep_newlines": keep_newlines},
        batched=True,
        remove_columns=wiki.column_names,
        num_proc=num_proc,
        desc="Tokenizing Wikipedia",
    )
    tokenized_owt = owt.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "keep_newlines": keep_newlines},
        batched=True,
        remove_columns=owt.column_names,
        num_proc=num_proc,
        desc="Tokenizing OpenWebText",
    )

    # Concatenate datasets before chunking so chunks can cross document boundaries within each source
    print(">> Concatenating tokenized corpora")
    tokenized_all = concatenate_datasets([tokenized_wiki, tokenized_owt])

    # Chunk into fixed-length blocks
    print(f">> Grouping into blocks of size {block_size}")
    lm_all = tokenized_all.map(
        group_texts,
        fn_kwargs={"block_size": block_size},
        batched=True,
        num_proc=num_proc,
        desc="Grouping text into fixed-length chunks",
    )

    # Filter out any empty batches (can happen if some map workers produced nothing)
    lm_all = lm_all.filter(lambda x: len(x["input_ids"]) == block_size)

    # Train/validation split
    print(">> Creating train/validation split (99/1)")
    lm_all = lm_all.shuffle(seed=seed)
    split = lm_all.train_test_split(test_size=0.01, seed=seed)
    lm_ds = DatasetDict(train=split["train"], validation=split["test"])

    # Save to disk (Hugging Face Dataset format)
    target = os.path.join(out_dir, f"wiki_owt_block{block_size}")
    print(">> Saving dataset to:", target)
    lm_ds.save_to_disk(target)

    # Quick stats
    n_train = len(lm_ds["train"])
    n_val = len(lm_ds["validation"])
    tokens_train = n_train * block_size
    tokens_val = n_val * block_size
    print("=== Summary ===")
    print(f"train examples:      {n_train}")
    print(f"validation examples: {n_val}")
    print(f"tokens (train):      {tokens_train:,}")
    print(f"tokens (valid):      {tokens_val:,}")
    print("Saved HF Dataset to:", target)
    print("You can now load it with `datasets.load_from_disk` and use DataCollatorForLanguageModeling for masking.")


def main():
    p = argparse.ArgumentParser(description="Preprocess Wikipedia + OpenWebText for LM pretraining")
    p.add_argument("--out_dir", type=str, default="training_data", help="Where to save the processed dataset")
    p.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="HF tokenizer name or path")
    p.add_argument("--block_size", type=int, default=128, help="Number of tokens per example")
    p.add_argument("--num_proc", type=int, default=4, help="Number of processes for map()")
    p.add_argument("--wiki_config", type=str, default="20220301.en", help="Wikipedia config (e.g., 20220301.en)")
    p.add_argument("--keep_newlines", action="store_true", help="Preserve newlines instead of replacing with spaces")
    p.add_argument("--max_wiki_samples", type=int, default=None, help="Optional cap for Wikipedia docs (for quick tests)")
    p.add_argument("--max_owt_samples", type=int, default=None, help="Optional cap for OpenWebText docs (for quick tests)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_and_save(
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
        block_size=args.block_size,
        num_proc=args.num_proc,
        wiki_config=args.wiki_config,
        keep_newlines=args.keep_newlines,
        max_wiki_samples=args.max_wiki_samples,
        max_owt_samples=args.max_owt_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()