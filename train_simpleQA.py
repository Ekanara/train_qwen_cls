# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, Trainer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import wandb
import os
import json
import evaluate


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_path = "./checkpoint-116000"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, model_max_length=2048)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


prompt = """ Bạn là một trợ lí tư vấn các vấn đề liên quan đến pháp luật. Hãy trả lời như một luật sư chuyên nghiệp. 

### Instruction:
{}

### Response"
{}"""
EOS_TOKEN = tokenizer.eos_token
def formatting_prompt_func(example):
    instructions =  example ['Heading']
    outputs = example['Content']
    texts=[]
    for instruction, output in zip(instructions, outputs):
        text = prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }

def load_json_dataset(train_path, val_path, test_path):
    dataset = load_dataset('json', data_files={
        'train': train_path,
        'validation': val_path,
        'test': test_path
    })
    dataset = dataset.map(formatting_prompt_func, batched=True)
    return dataset['train'], dataset['validation'], dataset['test']



train_path = r"VNLQA_80_10_10/VNLs1mpleQA_train.json"
val_path = r"VNLQA_80_10_10/VNLs1mpleQA_val.json"
test_path = r"VNLQA_80_10_10/VNLs1mpleQA_test.json"

# Load datasets
train_dataset, val_dataset, test_dataset = load_json_dataset(train_path, val_path, test_path)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


training_args = SFTConfig(
    output_dir="./sft_output_simpleQA",
    report_to="wandb",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500, 
    #save_steps=500, 
    eos_token="<|im_end|>",                      
    save_total_limit=5,                   
    load_best_model_at_end=True,          
    metric_for_best_model="eval_loss",
    weight_decay= 0.01,
    greater_is_better= False,
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model,
    dataset_text_field="text",
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length = 2048, 
    #compute_metrics=compute_metrics,
    #formatting_func=formatting_prompt_func,
    )

torch.cuda.empty_cache()

trainer.train()

metrics = trainer.evaluate()

print(metrics)

inference = pipeline(
    "text-generation",
    model=trainer.model,      
    tokenizer=trainer.tokenizer,
    max_seq_length=2048,
    device=0
)

# Load predictions
with open(r"Evaluate Results/VNLsimpleQA_test_predictions.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

predictions = [item["prediction"] for item in data]
references = [item["reference"] for item in data]

# Load metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# Compute ROUGE
rouge_result = rouge.compute(predictions=predictions, references=references)

# Compute BLEU
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# Compute BERTScore
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="vi")

# Print results
print("ROUGE:", rouge_result)
print("BLEU:", bleu_result)
print("BERTScore Precision:", sum(bertscore_result["precision"])/len(bertscore_result["precision"]))
print("BERTScore Recall:", sum(bertscore_result["recall"])/len(bertscore_result["recall"]))
print("BERTScore F1:", sum(bertscore_result["f1"])/len(bertscore_result["f1"]))
