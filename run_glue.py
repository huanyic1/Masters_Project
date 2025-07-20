#!/usr/bin/env python
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default=None, metadata={"help": "Task name: " + ", ".join(task_to_keys.keys())})
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys:
                raise ValueError("Unknown task, choose from: " + ", ".join(task_to_keys))
        elif self.dataset_name is None and (self.train_file is None or self.validation_file is None):
            raise ValueError("Need a GLUE task, a training/validation file, or a dataset name.")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path or identifier of the pretrained model"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if is_main_process():
        logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    if data_args.task_name is not None:
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir, token=model_args.token)
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        if training_args.do_predict and data_args.test_file:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    is_regression = data_args.task_name == "stsb" if data_args.task_name else raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    label_list = raw_datasets["train"].features["label"].names if data_args.task_name and not is_regression else raw_datasets["train"].unique("label")
    label_list.sort() if not is_regression else None
    num_labels = 1 if is_regression else len(label_list)


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Always pull tokenizer from Hugging Face hub, not from local path
    hf_tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(
        hf_tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )


    sentence1_key, sentence2_key = task_to_keys.get(data_args.task_name, ("sentence1", "sentence2")) if data_args.task_name else (None, None)

    def preprocess_function(examples):
        args = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*args, padding="max_length" if data_args.pad_to_max_length else False, max_length=data_args.max_seq_length, truncation=True)
        if "label" in examples and not is_regression:
            result["label"] = [label_list.index(l) if isinstance(l, str) else l for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="Tokenizing dataset"):
        raw_datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    def compute_metrics(p: EvalPrediction):
        preds = np.squeeze(p.predictions) if is_regression else np.argmax(p.predictions, axis=1)
        return evaluate.load("glue", data_args.task_name).compute(predictions=preds, references=p.label_ids)


    if training_args.do_eval:
        if data_args.task_name == "mnli":
            eval_dataset = raw_datasets["validation_matched"]
        else:
            eval_dataset = raw_datasets["validation"]
    else:
        eval_dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator if data_args.pad_to_max_length else DataCollatorWithPadding(tokenizer),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint or last_checkpoint)
        trainer.save_model()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and "test" in raw_datasets:
        logger.info("*** Predict ***")
        predictions = trainer.predict(raw_datasets["test"], metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()


"""
  ***** SST our trained bert eval metrics ***** 
  epoch                   =        3.0
  eval_accuracy           =     0.9106
  eval_loss               =     0.3059
  eval_runtime            = 0:00:02.77
  eval_samples_per_second =    314.252
  eval_steps_per_second   =      2.523

  ***** CoLA our BERT eval metrics *****
  epoch                     =        3.0
  eval_loss                 =     0.5487
  eval_matthews_correlation =     0.4802
  eval_runtime              = 0:00:01.87
  eval_samples_per_second   =    557.413
  eval_steps_per_second     =       4.81

  ***** mnli our BERT eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =     0.8207
  eval_loss               =     0.4937
  eval_runtime            = 0:00:35.49
  eval_samples_per_second =    276.531
  eval_steps_per_second   =      2.169

  ***** mrpc our BERT eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =     0.8407
  eval_f1                 =      0.887
  eval_loss               =     0.3684
  eval_runtime            = 0:00:00.69
  eval_samples_per_second =     582.96
  eval_steps_per_second   =      5.715

  ***** qnli our BERT eval metrics *****
  eval_accuracy           =     0.9017
  eval_loss               =     0.2728
  eval_runtime            = 0:00:19.59
  eval_samples_per_second =    278.837
  eval_steps_per_second   =      8.728
  
  ***** qqp our BERT eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =     0.9008
  eval_f1                 =     0.8662
  eval_loss               =     0.2539
  eval_runtime            = 0:02:16.04
  eval_samples_per_second =    297.187
  eval_steps_per_second   =      2.323

  ***** rte our BERT eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =     0.5993
  eval_loss               =     0.6698
  eval_runtime            = 0:00:00.54
  eval_samples_per_second =     511.82
  eval_steps_per_second   =      5.543

  ***** stsb our BERT eval metrics *****
  epoch                   =        3.0
  eval_loss               =     0.6497
  eval_pearson            =     0.8534
  eval_runtime            = 0:00:03.09
  eval_samples_per_second =    484.534
  eval_spearmanr          =     0.8533
  eval_steps_per_second   =      3.876

  ***** wnli our BERT eval metrics ***** 
  epoch                   =        3.0
  eval_accuracy           =     0.2817
  eval_loss               =     0.7241
  eval_runtime            = 0:00:00.37
  eval_samples_per_second =    188.391
  eval_steps_per_second   =       7.96
  
  ***** SST bert-base-uncased eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =      0.922
  eval_loss               =     0.2667
  eval_runtime            = 0:00:03.01
  eval_samples_per_second =    288.882
  eval_steps_per_second   =      2.319

  ***** bert base Cola eval metrics ***** 1 core
  epoch                     =        3.0
  eval_loss                 =     0.5202
  eval_matthews_correlation =     0.5779
  eval_runtime              = 0:00:02.55
  eval_samples_per_second   =    407.495
  eval_steps_per_second     =     12.893

  ***** eval metrics ***** 4 cores
  epoch                     =        3.0
  eval_loss                 =     0.4929
  eval_matthews_correlation =     0.5393
  eval_runtime              = 0:00:01.65
  eval_samples_per_second   =    631.538
  eval_steps_per_second     =       5.45

"""

