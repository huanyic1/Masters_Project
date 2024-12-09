import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

from resprop_linear import resprofify_bert
from plot import plot_log_histories


# Load Model
model_name = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={
        0: "Negative",
        1: "Positive"
    }
)

# Load Data
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

dataset = load_dataset("fancyzhx/yelp_polarity")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(64000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Accuracy
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


log_histories = {}
for reuse_percentage in [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]:
    model = resprofify_bert(base_model, reuse_percentage=reuse_percentage)
    training_args = TrainingArguments(
        output_dir=f"trainer_out/{model_name.replace('/', '-')}/rp_{int(reuse_percentage * 100)}",
        eval_strategy="steps",
        num_train_epochs=4,
        eval_steps=1/64
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    log_histories[reuse_percentage] = trainer.state.log_history


# Generate Plot
plot_log_histories(log_histories, file_name="result.png")
