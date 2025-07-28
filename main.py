from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from schemas.news import NewsInput
# Import language utilities from utils.fact_check as they are now self-contained there
from utils.fact_check import detect_language, map_language_code, perform_agent_fact_check
# from utils.fact_check import fetch_fact_check_results, decide_final_verdict # REMOVE these imports
from utils.predictor import fake_news_predict
from utils.logger import log_prediction
from models.loader import models # Assuming this loads your ML models

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
        
        # Use the detect_language from utils.fact_check
        lang_detected = detect_language(text)
        lang_code = map_language_code(lang_detected) # Use map_language_code from utils.fact_check

        # Your existing ML model prediction
        label, confidence = fake_news_predict(text, lang_code, models)
        print(f"ML Model Prediction: Label={label}, Confidence={confidence:.2f}")

        # --- NEW: Call the agent-based fact-checking ---
        # This function now handles all the external API calls and the final verdict logic
        agent_fact_check_result = await perform_agent_fact_check(text, lang_code)
        
        final_verdict = agent_fact_check_result["final_verdict"]
        explanation = agent_fact_check_result["explanation"] # The detailed explanation from the agent
        fact_check_status = agent_fact_check_result["fact_check_status"]
        fact_check_sources = agent_fact_check_result["fact_check_sources"] # This will be empty or parsed from agent output

        # Logging the result to CSV
        log_entry = {
            "text": text,
            "detected_language": lang_detected,
            "model_language_used": lang_code,
            "model_prediction": label, # From your ML model
            "model_confidence": confidence, # From your ML model
            "fact_check_status": fact_check_status, # From agent
            "final_verdict": final_verdict, # From agent
            "timestamp": datetime.utcnow().isoformat()
        }

        log_prediction(log_entry)

        # ✅ Final API response
        return {
            "detected_language": lang_detected,
            "model_language_used": lang_code,
            "ml_prediction": label,
            "confidence": confidence,
            "fact_check_status": fact_check_status,
            "fact_check_sources": fact_check_sources, # Note: This will be empty as per utils/fact_check.py current design
            "final_verdict": final_verdict,
            "agent_explanation": explanation # New field to include the agent's detailed explanation
        }

    except Exception as e:
        print(f"Error in predict_news endpoint: {e}")
        return {"error": str(e)}