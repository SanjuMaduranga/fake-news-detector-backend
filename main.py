from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from schemas.news import NewsInput
from utils.language import detect_language, map_language_code
from utils.predictor import fake_news_predict
from utils.fact_check import fetch_fact_check_results, decide_final_verdict
from utils.logger import log_prediction
from models.loader import models

app = FastAPI()

# CORS config for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_news(news: NewsInput):
    try:
        text = news.text.strip()
        lang_detected = detect_language(text)
        lang_code = map_language_code(lang_detected)

        label, confidence = fake_news_predict(text, lang_code, models)

        # Fetch fact-check data from Google API
        fact_check_results = await fetch_fact_check_results(text)
        fact_checks = []
        for claim in fact_check_results:
            claim_review = claim.get("claimReview", [{}])[0]
            fact_checks.append({
                "claim": claim.get("text", ""),
                "review_text": claim_review.get("text", ""),
                "publisher": claim_review.get("publisher", {}).get("name", ""),
                "url": claim_review.get("url", "")
            })

        # Use model + fact-check info to decide verdict
        final_verdict = decide_final_verdict(label, confidence, fact_checks)

        # Logging the result to CSV
        log_entry = {
            "text": text,
            "detected_language": lang_detected,
            "model_language_used": lang_code,
            "model_prediction": label,
            "model_confidence": confidence,
            "fact_check_status": "Found" if fact_checks else "Not found",
            "final_verdict": final_verdict,
            "timestamp": datetime.utcnow().isoformat()
        }

        log_prediction(log_entry)

        # ✅ Final API response
        return {
            "detected_language": lang_detected,
            "model_language_used": lang_code,
            "ml_prediction": label,
            "confidence": confidence,
            "fact_check_status": "Found" if fact_checks else "Not found",
            "fact_check_sources": fact_checks,
            "final_verdict": final_verdict
        }

    except Exception as e:
        return {"error": str(e)}