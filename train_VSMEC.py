from datasets import load_dataset,  DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
import evaluate
import numpy as np

if __name__ =='__main__':

    dataset = load_dataset("tmnam20/ViGLUE",name='vsmec')
    #dataset.pop('validation')
    dataset.pop("test",None)  # Drop the test set
    # split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # dataset = DatasetDict({
    #     "train": split_dataset['train'],
    #     "validation": split_dataset['test']
    # })
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    model_name = "./checkpoint-116000"  
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #print(train_dataset[2])
    label_map = {"entailment": 0, "not_entailment": 1}  # Convert to numerical labels

    def format_prompt(sentence, raw_sentence,label = None):
        prompt = f"<Sentence>: {sentence} </Sentence>\n <Raw_Sentence>: {raw_sentence} </Raw_Sentence>"  # Convert sentence to lowercase
        return prompt
    #formatted_text = format_prompt("Ai đã lật ngược phán quyết của Taft Vale?", "Một trong những hành động đầu tiên của Chính phủ Tự do mới là đảo ngược phán quyết của Taff Vale.", 'entailment')
    #print(formatted_text)

    def preprocess_function(examples):
        texts = [format_prompt(s, r) for s, r in zip(examples["sentence"], examples["raw_sentence"])]
        tokenized = tokenizer(texts, truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "label": examples["label"]  # Overwrite "label"
        }

    dataset = dataset.map(preprocess_function, batched=True)
    #encoded_dataset.save_to_disk("qnli_SLM_preprocessed")
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {
        0: "anger",
        1: "disgust",
        2: "enjoyment",
        3: "fear",
        4: "other",
        5: "sadness",
        6: "surprise"
    }

    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label), trust_remote_code=True, label2id = label2id, id2label=id2label)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir="Qwenv2.5_VSMEC_results",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    

