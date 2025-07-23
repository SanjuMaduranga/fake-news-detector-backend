import torch

def fake_news_predict(text: str, lang_code: str, models):
    if lang_code not in models:
        return "Unsupported Language", 0.0
    tokenizer = models[lang_code]["tokenizer"]
    model = models[lang_code]["model"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Real" if pred == 1 else "Fake"
    return label, round(confidence, 3)
