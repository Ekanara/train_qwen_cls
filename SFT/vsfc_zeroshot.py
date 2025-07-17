# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from trl import SFTTrainer
import torch
import pandas as pd

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model_path = r"./checkpoint-116000"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

train_path = r"./checkpoint-116000/vsfc/train-00000-of-00001.parquet"
val_path = r"./checkpoint-116000/vsfc/validation-00000-of-00001.parquet"
test_path = r"./checkpoint-116000/vsfc/test-00000-of-00001.parquet"

# Load parquet files
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def formatting_prompt_func_VSFC(example):
    return ("<Instruct>Classify the sentiment of the given sentence into 3 classes: "
	"(0: Negative, 1: Neutral, 2: Positive)</Instruct>\n"
	f"<Sentence>{example['sentence']}</Sentence>\n"
	f"<Answer>{example['label']}</Answer>"
	)

# Training Arguments
training_args = TrainingArguments(
    output_dir="VSFC_zeroshot",
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
    push_to_hub=False,
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    formatting_func=formatting_prompt_func_VSFC,
)

trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)
