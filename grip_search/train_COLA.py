from datasets import load_dataset,  DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
import evaluate
import numpy as np


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        #"per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [8, 16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.3),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.2),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "polynomial"]),
        "optim": trial.suggest_categorical("optim", ["adamw_torch", "adafactor"]),
    }



if __name__ =='__main__':

    dataset_train = load_dataset("tmnam20/ViGLUE", split='train', name='cola')
    #dataset.pop("test",None)  # Drop the test set
    dataset_val = load_dataset("tmnam20/ViGLUE", split='validation', name='cola')
    # split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # dataset = DatasetDict({
    #     "train": split_dataset['train'],
    #     "validation": split_dataset['test']
    # })
    dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val
    })
    # train_dataset = dataset["train"]
    # val_dataset = dataset["validation"]

    model_name = "./checkpoint-116000"  
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    #model.config.pad_token_id = model.config.eos_token_id
    #label_map = {"unacceptable": 0, "acceptable": 1}  # Convert to numerical labels

    def format_prompt(sentence, label = None):
        prompt = f"<Sentence>: {sentence} </Sentence>"  # Convert sentence to lowercase
        return prompt
    #formatted_text = format_prompt("Ai đã lật ngược phán quyết của Taft Vale?", "Một trong những hành động đầu tiên của Chính phủ Tự do mới là đảo ngược phán quyết của Taff Vale.", 'entailment')
    #print(formatted_text)

    def preprocess_function(examples):
        texts = [format_prompt(s) for s in examples["sentence"]]
        tokenized = tokenizer(texts, truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "label": examples["label"]  # Overwrite "label"
        }

    dataset = dataset.map(preprocess_function, batched=True)
    #encoded_dataset.save_to_disk("qnli_SLM_preprocessed")
    #accuracy = evaluate.load("accuracy")
    mcc = evaluate.load("matthews_correlation")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        #result = accuracy.compute(predictions=predictions, references=labels)
        result = mcc.compute(predictions=predictions, references=labels)
        return result if result is not None else {"matthews_correlation": 0.0}

    id2label = {0: "unacceptable", 1: "acceptable"}
    label2id = {"unacceptable": 0, "acceptable": 1}
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, trust_remote_code=True, label2id = label2id, id2label=id2label)
        model.config.pad_token_id = model.config.eos_token_id
        return model
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir="Qwenv2.5_COLA_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        lr_scheduler_type="cosine",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=3,
        # bf16=True,  
        bf16=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        # use_liger_kernel=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #def compute_objective(metrics):
    #    return metrics["eval_accuracy"]
    
    def compute_objective(metrics):
        return metrics["eval_matthews_correlation"]
    
    best_trials = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=50,
        compute_objective=compute_objective,
    )

    best_training_args = TrainingArguments(
        output_dir="Qwenv2.5_COLA_results_mcc",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        #lr_scheduler_type="cosine",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        #gradient_accumulation_steps=2,
        num_train_epochs=3,
        #weight_decay=0.01,
        save_total_limit=1,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_matthews_correlation",
        greater_is_better=True,
        push_to_hub=False,
        **best_trials.hyperparameters  # Apply the best hyperparameters
    )

    # best_training_args = TrainingArguments(
    #     output_dir="Qwenv2.5_QNLI_results",
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     logging_strategy="epoch",
    #     lr_scheduler_type="cosine",
    #     per_device_eval_batch_size=3,
    #     gradient_accumulation_steps=2,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     save_total_limit=3,
    #     fp16=True,
    #     load_best_model_at_end=True,
    #     push_to_hub=False,
    #     **best_trials.hyperparameters  # Apply the best hyperparameters
    # )

    trainer = Trainer(
        model_init=model_init,
        args=best_training_args,  # Use the updated training arguments
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.push_to_hub("Ekanari/Qwenv2.5_COLA_results")
    tokenizer.push_to_hub("Ekanari/Qwenv2.5_COLA_results")
    # best_model.push_to_hub("presencesw/Qwen2.5_COLA_results")
    # tokenizer.push_to_hub("presencesw/Qwen2.5_COLA_results")
    # print(best_trials)
    # trainer.train()
