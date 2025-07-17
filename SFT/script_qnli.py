from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, Trainer, pipeline
from datasets import load_dataset,  DatasetDict
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import wandb
import os
import numpy as np
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()else "cpu"
    #print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")
    model_path = "./checkpoint-116000"
    #model_path = "vinai/PhoGPT-4B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=1024)
    #os.environ["WANDB_PROJECT"]=""
    #os.environ["WANDB_ENTITY"] = ""
    #dataset = load_dataset("parquet", data_files={"train": "train_qnli."})
    dataset = load_dataset('tmnam20/ViGLUE','qnli', split='train')
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": split_dataset['train'],
        "validation": split_dataset['test']
    })
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]	 
    #eval_dataset = load_dataset('tmnam20/ViGLUE','qnli', split='validation')
    def formatting_prompt_func(example):
        return (
            "<Instruction>Câu sau có trả lời cho câu hỏi không?</Instruction>\n"
            "<Input>Câu hỏi: {}\nCâu: {}</Input>\n"
            "<Output>{}</Output>\n"
        ).format(example["question"], example["sentence"], example["label"])
    train_dataset, eval_dataset = train_dataset.map(lambda x: {"text": formatting_prompt_func(x)}), eval_dataset.map(lambda x: {"text": formatting_prompt_func(x)})
    training_args = SFTConfig(
        output_dir="./qwen_sft_output_qnli",
        report_to=["wandb"],
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
	#eval_steps=1,
        save_strategy="epoch",
        #logging_strategy="epoch",
        #logging_steps=1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,
    )
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda x: x["text"]
    )
    trainer.train()
    trainer.save_model("./qwen_sft_output_qnli")
    tokenizer.save_pretrained("./qwen_sft_output_qnli")
    print("Training complete and model saved to ./qwen_sft_output_qnli")
if __name__ == "__main__":
    main()
