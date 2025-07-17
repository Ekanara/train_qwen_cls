from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from trl import SFTTrainer
import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"./checkpoint-116000"

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

train_df = pd.read_parquet(r"./checkpoint-116000/vtoc/train-00000-of-00001.parquet")

train_split_df, test_split_df = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    stratify=train_df['label']
)
val_df = pd.read_parquet(r"./checkpoint-116000/vtoc/train-00000-of-00001.parquet")

train_dataset = Dataset.from_pandas(train_split_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_split_df)

def formatting_prompt_func_VTOC_cls(example):
    return (
        "<Instruct>Classify the sentence into one of 15 classes (0 to 14).</Instruct>\n"
        f"<Sentence>{example['sentence']}</Sentence>\n"
        f"<Answer>{example['label']}</Answer>"
    )

# Training Arguments
training_args = TrainingArguments(
    output_dir="VTOC_zeroshot",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_prompt_func_VTOC_cls,
)
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)
