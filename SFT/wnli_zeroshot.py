# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset,  DatasetDict, Dataset
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

dataset = load_dataset('tmnam20/ViGLUE','wnli', split='train')
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

dataset = DatasetDict({
"train": split_dataset['train'],
"validation": split_dataset['test']
})

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

def formatting_prompt_func_VMNLI(example):
    return (
        "<Instruct>Determine the relationship between the sentence1 and sentence2. "
        "Labels: 0 = not_entailment, 1 = entailment.</Instruct>\n"
        f"<Sentence1>{example['sentence1']}</Sentence1>\n"
        f"<Sentence2>{example['sentence2']}</Sentence2>\n"
        f"<Answer>{example['label']}</Answer>"
    )

# Training Arguments
training_args = TrainingArguments(
    output_dir="wnli_zeroshot",
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
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompt_func_VMNLI,
)

trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)
