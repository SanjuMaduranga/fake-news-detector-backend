from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_paths = {
    "si": "sanjulamaduranga/Sinhala_Fake_News_Detection",
    "en": "sanjulamaduranga/Fake_News_Detection",
    "ta": "sanjulamaduranga/Tamil_Fake_News_Detection"
}

models = {}
for lang_code, repo in model_paths.items():
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    models[lang_code] = {
        "tokenizer": tokenizer,
        "model": model
    }
