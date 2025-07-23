import os
from dotenv import load_dotenv
import httpx

load_dotenv()

GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


# API 1: Google Fact Check
async def fetch_google_fact_check(query: str):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": GOOGLE_FACT_CHECK_API_KEY}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("claims", [])
        return []


# API 2: NewsAPI (Fallback)
async def fetch_newsapi_articles(query: str):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{"claim": article["title"], "review_text": article["description"], "publisher": article["source"]["name"]}
                    for article in data.get("articles", [])]
        return []


# API 3: GNews (Fallback) 
async def fetch_gnews_articles(query: str):
    url = f"https://gnews.io/api/v4/search"
    params = {"q": query, "token": GNEWS_API_KEY, "lang": "en"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [{"claim": article["title"], "review_text": article["description"], "publisher": article["source"]["name"]}
                    for article in data.get("articles", [])]
        return []


# Combined Fallback Chain
async def fetch_fact_check_results(query: str):
    results = await fetch_google_fact_check(query)
    if results:
        return results

    print("Google API returned no claims. Trying NewsAPI...")
    results = await fetch_newsapi_articles(query)
    if results:
        return results

    print("NewsAPI returned no claims. Trying GNews API...")
    return await fetch_gnews_articles(query)


# Final Verdict Logic 
def decide_final_verdict(model_label: str, model_confidence: float, fact_checks: list) -> str:
    trusted_publishers = {
        # English/International
        "BBC": 1, "Reuters": 1, "AP News": 1, "USA Today": 1,
        "FactCheck.org": 1, "Full Fact": 1, "Snopes": 1,
        "Science Feedback": 1, "AAP": 1, "PolitiFact": 1,
        "The Associated Press": 1,

        # Sinhala
        "FactCheck.lk": 1, "Hashtag Generation": 1, "Ada derana": 1,
        "Groundviews": 1, "Sri Lanka FactCheck": 1,

        # Tamil
        "BOOM Tamil": 1, "Fact Crescendo Tamil": 1, "Vishvas News Tamil": 1,
        "Youturn.in": 1, "Newschecker Tamil": 1
    }

    fake_keywords = ["false", "fake", "misleading", "not true", "debunked", "incorrect"]
    real_keywords = ["true", "correct", "accurate", "verified", "supported"]

    trusted_real = 0
    trusted_fake = 0

    for review in fact_checks:
        text = (review.get("review_text", "") + " " + review.get("claim", "")).lower()
        publisher = review.get("publisher", "").strip()

        if publisher in trusted_publishers:
            if any(word in text for word in real_keywords):
                trusted_real += 1
            elif any(word in text for word in fake_keywords):
                trusted_fake += 1

    # Final Logic
    if trusted_real >= 2 and model_label == "Real":
        return "Real"
    elif trusted_fake >= 2 and model_label == "Fake":
        return "Fake"
    elif trusted_fake >= 1 and model_label == "Real":
        return "Uncertain"
    elif model_confidence < 0.6:
        return "Uncertain"
    return model_label