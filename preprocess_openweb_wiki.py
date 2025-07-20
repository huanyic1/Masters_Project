#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain

def preprocess_and_cache(
    dataset_name: str,
    config_name: str,
    split: str,
    tokenizer: AutoTokenizer,
    block_size: int,
    cache_dir: str,
):
    """
    - Loads `dataset_name` + `config_name` on `split`
    - Tokenizes the `text` column
    - Concatenates & chunks into blocks of `block_size`
    - Saves to `cache_dir/{dataset_name}-{config_name}-{split}`
    """
    # 1) Load raw
    ds = load_dataset(dataset_name, config_name, split=split)

    # 2) Tokenize (return overflowing tokens so we can regroup manually)
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            return_overflowing_tokens=True,
            return_length=True,
            padding="max_length",
        )

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
    )

    # 3) Group into exact block_size chunks
    def group_texts(examples):
        # flatten out all overflowing chunks
        all_ids = list(chain.from_iterable(examples["input_ids"]))
        # drop remainder to make divisible
        total_len = (len(all_ids) // block_size) * block_size
        all_ids = all_ids[:total_len]
        # re-chunk
        chunks = [
            all_ids[i : i + block_size]
            for i in range(0, total_len, block_size)
        ]
        return {"input_ids": chunks, "attention_mask": [[1]*block_size]*len(chunks)}

   
    grouped = tokenized.map(
            group_texts,
            batched=True,
            remove_columns=tokenized.column_names,
            num_proc=args.num_proc,
    )

    # 4) Save
    out_path = os.path.join(cache_dir, f"{dataset_name}-{config_name}-{split}")
    os.makedirs(out_path, exist_ok=True)
    grouped.save_to_disk(out_path)
    print(f"✓ Saved {dataset_name}/{config_name}/{split} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Wiki + OpenWebText into token blocks"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HF tokenizer to use",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Length of each token block",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cached_datasets",
        help="Where to write processed splits",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for `dataset.map(...)`",
    )
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Wikipedia 20200501.en, train split
    preprocess_and_cache(
        dataset_name="wikipedia",
        config_name="20220301.en",
        split="train",
        tokenizer=tok,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
    )

    # OpenWebText, train split
    preprocess_and_cache(
        dataset_name="openwebtext",
        config_name=None,
        split="train",
        tokenizer=tok,
        block_size=args.block_size,
        cache_dir=args.cache_dir,
    )