import evaluate

# Load predictions
with open(r"Evaluate Results\VNLsimpleQA_test_predictions.json", "r", encoding="utf-8-sig") as f:
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
