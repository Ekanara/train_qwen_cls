from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import torch
from trl import SFTTrainer
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"./checkpoint-116000"

# Load model + tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

train_path = r"./checkpoint-116000/vsmec/train-00000-of-00001.parquet"
val_path = r"./checkpoint-116000/vsmec/validation-00000-of-00001.parquet"
test_path = r"./checkpoint-116000/vsmec/test-00000-of-00001.parquet"

train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def formatting_prompt_func_VSMEC(example):
    return (
        "<Instruct>Classify the emotion of the given sentence into 7 classes:\n"
        "(0: Anger, 1: Disgust, 2: Enjoyment, 3: Fear, 4: Other, 5: Sadness, 6: Surprise)</Instruct>\n"
        f"<Sentence>{example['sentence']}</Sentence>\n"
        f"<Answer>{example['label']}</Answer>"
    )

training_args = TrainingArguments(
    output_dir="VSMEC_zeroshot",
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
    bf16=True,  # Nếu GPU không hỗ trợ có thể chuyển thành fp16=True hoặc False
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
    formatting_func=formatting_prompt_func_VSMEC,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
