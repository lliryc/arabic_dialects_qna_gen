from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

def get_qnq_log():
    records = []
    with open("question_logs_egyptian.txt", "r") as log_in:
        log_content = log_in.read()
        log_strs = log_content.split("\n\n")
        for log_str in log_strs:
            try:
                log_array = json.loads(log_str)
                complexities = ["Challenging", "Moderate", "Easy"]
                for log, complexity in zip(log_array, complexities):
                    record = {}
                    record["Question"] = log["Question"]
                    record["Answer"] = log["Answer"]
                    record["LLMQuestionDifficulty"] = complexity
                    records.append(record)
            except Exception as e:
                print(e)
                continue
    return records


if __name__ == "__main__":

    records_logs = get_qnq_log()
    model_name = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    for records_log in records_logs:
        text = records_log["Question"]
        if text is None:
            continue
        inputs = tokenizer(text, return_tensors="pt")

        # Run inference
        with torch.inference_mode():
            logits = model(**inputs).logits

        # Get probabilities using softmax
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get top 3 predictions
        top1_probs, top1_indices = torch.topk(probabilities, 1, dim=-1)
        
        dialects = []

        for i in range(1):
            pred_idx = top1_indices[0][i].item()
            confidence = top1_probs[0][i].item()
            dialect = model.config.id2label[pred_idx]
            dialects.append((dialect, confidence))
        print(f"{dialects[0][0]}, {records_log["LLMQuestionDifficulty"]}")
