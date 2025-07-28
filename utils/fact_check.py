import os
from dotenv import load_dotenv
import httpx
import asyncio
import re
from langdetect import detect, DetectorFactory
import google.generativeai as genai
import langid # Using langid for potentially better language detection
import requests # For making HTTP requests
from bs4 import BeautifulSoup # For parsing HTML

# Removed LangChain/LangGraph imports as agents are no longer used
# No longer importing DuckDuckGoSearchRun or any other LangChain tool


# --- Configuration and Environment Setup ---
# Ensure reproducible results for langdetect (optional, but good practice)
DetectorFactory.seed = 0

# Load environment variables from .env file (important for this module)
load_dotenv()

# API Keys (ensure these are in your .env file)
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # For Google's Gemini LLM

# Removed GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID

# Configure Gemini for direct API calls
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_direct_model = genai.GenerativeModel('gemini-1.5-flash')

# No longer initializing DuckDuckGoSearchRun tool


# --- API Fetching Functions (Existing ones) ---

async def fetch_google_fact_check(query: str):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": GOOGLE_FACT_CHECK_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status() # Raise an exception for HTTP errors
            data = response.json()
            return data.get("claims", [])
        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching Google Fact Check: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"Request error fetching Google Fact Check: {e}")
            return []

async def fetch_newsapi_articles(query: str, language_code: str = "en"):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": language_code}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{"claim": article["title"], "review_text": article["description"], "publisher": article["source"]["name"], "url": article["url"]}
                            for article in data.get("articles", [])]
        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching NewsAPI ({language_code}): {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"Request error fetching NewsAPI ({language_code}): {e}")
            return []

async def fetch_gnews_articles(query: str, language_code: str = "en"):
    url = f"https://gnews.io/api/v4/search"
    params = {"q": query, "token": GNEWS_API_KEY, "lang": language_code}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{"claim": article["title"], "review_text": article["description"], "publisher": article["source"]["name"], "url": article["url"]}
                            for article in data.get("articles", [])]
        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching GNews ({language_code}): {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"Request error fetching GNews ({language_code}): {e}")
            return []

# Removed fetch_google_custom_search function.


# --- NEW: Web Scraping Function for Search Results ---
async def scrape_search_results(query: str, language_code: str = "en"):
    """
    Performs a search on DuckDuckGo and scrapes the results using requests and BeautifulSoup.
    Returns a list of dictionaries with 'claim', 'review_text', 'publisher', 'url'.
    """
    search_url = f"https://duckduckgo.com/html/?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"Scraping DuckDuckGo for: '{query}' (Lang: {language_code})")
    try:
        # Use asyncio.to_thread for synchronous requests.get in an async function
        response = await asyncio.to_thread(requests.get, search_url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        # DuckDuckGo HTML structure for results might look like this (can change!):
        # <div class="result__body">
        #   <h2 class="result__title"><a>...</a></h2>
        #   <a class="result__url">...</a>
        #   <div class="result__snippet">...</div>
        # </div>
        
        # Find all search result containers
        for result_div in soup.find_all('div', class_='result'):
            title_tag = result_div.find('h2', class_='result__title')
            url_tag = result_div.find('a', class_='result__url')
            snippet_div = result_div.find('div', class_='result__snippet')

            title = title_tag.text.strip() if title_tag else "No Title"
            url = url_tag['href'] if url_tag and 'href' in url_tag.attrs else "No URL"
            snippet = snippet_div.text.strip() if snippet_div else "No Snippet"

            # Filter out internal DuckDuckGo links if desired
            if "duckduckgo.com" not in url:
                results.append({
                    "claim": title,
                    "review_text": snippet,
                    "publisher": "DuckDuckGo Scrape", # Indicate source of this data
                    "url": url
                })
        
        print(f"Scraped {len(results)} results from DuckDuckGo.")
        return results

    except requests.exceptions.RequestException as e:
        print(f"Request error during scraping DuckDuckGo: {e}")
        return []
    except Exception as e:
        print(f"Error during parsing DuckDuckGo results: {e}")
        return []


# --- Language Detection (using langid for broader support) ---
def detect_language(text: str) -> str:
    """Detects the language of the input text using langid, with specific checks for Sinhala/Tamil Unicode ranges."""
    try:
        lang, _ = langid.classify(text)
        
        if any('\u0D80' <= ch <= '\u0DFF' for ch in text): # Sinhala Unicode range
            return "si"
        elif any('\u0B80' <= ch <= '\u0BFF' for ch in text): # Tamil Unicode range
            return "ta"
        
        return lang
    except Exception as e:
        print(f"Error in language detection: {e}. Defaulting to 'en'.")
        return "en"

def map_language_code(code: str) -> str:
    """Maps detected language code to a simplified code if necessary."""
    if code.startswith("si"):
        return "si"
    elif code.startswith("ta"):
        return "ta"
    elif code.startswith("en"):
        return "en"
    return "en" # Default to English if no specific mapping


# --- Final Verdict Logic ---
TRUSTED_PUBLISHERS = {
    "BBC": 1, "Reuters": 1, "AP News": 1, "USA Today": 1,
    "FactCheck.org": 1, "Full Fact": 1, "Snopes": 1,
    "Science Feedback": 1, "AAP": 1, "PolitiFact": 1,
    "The Associated Press": 1, "Associated Press": 1,

    "FactCheck.lk": 1, "Hashtag Generation": 1, "Ada Derana": 1,
    "Groundviews": 1, "Sri Lanka FactCheck": 1, "News First": 1, "Derana": 1,
    "Mawbima": 1, "Lankadeepa": 1, "Divaina":1, "Silumina":1,

    "BOOM Tamil": 1, "Fact Crescendo Tamil": 1, "Vishvas News Tamil": 1,
    "Youturn.in": 1, "Newschecker Tamil": 1, "Puthiya Thalaimurai": 1,
    "Daily Thanthi": 1, "Hindu Tamil": 1, "Dinamalar": 1, "Virakesari":1,
}

VERDICT_KEYWORDS = {
    "en": {
        "fake": ["false", "fake", "misleading", "not true", "debunked", "incorrect", "hoax", "disinformation", "untrue"],
        "real": ["true", "correct", "accurate", "verified", "supported", "fact", "real", "authentic"],
        "uncertain": ["unproven", "unverified", "insufficient evidence", "disputed", "needs more info"]
    },
    "si": {
        "fake": ["බොරු", "ව්‍යාජ", "වැරදි", "මිත්‍යා", "පදනම් විරහිත", "අසත්‍ය", "කළ", "නොවේ", "නොමඟ යවන"],
        "real": ["සැබෑ", "නිවැරදි", "තහවුරු කරන ලද", "සත්‍ය", "සහතික", "වාර්තාව", "කිසිදු වෙනසක් නැත"],
        "uncertain": ["අවිනිශ්චිත", "තහවුරු නොකළ", "සාක්ෂි නොමැති", "විවාදයට තුඩු දුන්"]
    },
    "ta": {
        "fake": ["பொய்", "போலி", "தவறான", "நம்பிக்கையற்ற", "சரியில்லாத", "ஏமாற்று", "உண்மையல்ல"],
        "real": ["உண்மை", "சரியானது", "உறுதிப்படுத்தப்பட்ட", "சரிபார்க்கப்பட்டது", "உறுதி", "சரியானது"],
        "uncertain": ["உறுதியற்ற", "நிரூபிக்கப்படாத", "போதுமான ஆதாரங்கள் இல்லை", "சந்தேகத்திற்குரிய"]
    }
}

def decide_final_verdict(model_label: str, model_confidence: float, fact_checks: list, detected_lang: str) -> str:
    """
    Combines model output with external fact-checks to determine a final verdict.
    """
    trusted_real = 0
    trusted_fake = 0
    trusted_uncertain = 0
    
    any_search_results_found = bool(fact_checks)

    for review in fact_checks:
        text = (review.get("review_text", "") + " " + review.get("claim", "")).lower()
        publisher = review.get("publisher", "").strip()
        
        current_keywords = VERDICT_KEYWORDS.get(detected_lang, VERDICT_KEYWORDS["en"])

        veracity_found = "unknown"
        if any(word in text for word in current_keywords["fake"]):
            veracity_found = "fake"
        elif any(word in text for word in current_keywords["real"]):
            veracity_found = "real"
        elif any(word in text for word in current_keywords["uncertain"]):
            veracity_found = "uncertain"
        
        if publisher in TRUSTED_PUBLISHERS:
            if veracity_found == "real":
                trusted_real += 1
            elif veracity_found == "fake":
                trusted_fake += 1
            elif veracity_found == "uncertain":
                trusted_uncertain += 1

    # Final Logic: Prioritize trusted sources
    if trusted_real >= 1 and trusted_fake >= 1: # Conflicting trusted sources
        return "Uncertain"
    if trusted_real >= 2:
        return "Real"
    if trusted_fake >= 2:
        return "Fake"
    
    # If a single trusted source has a strong opposing view, it creates uncertainty
    if trusted_fake == 1 and model_label == "Real":
        return "Uncertain"
    if trusted_real == 1 and model_label == "Fake":
        return "Uncertain"
        
    # If no strong consensus from trusted sources, and some trusted 'uncertain' results
    if trusted_uncertain > 0 and (trusted_real == 0 and trusted_fake == 0):
        return "Uncertain"

    # If initial model confidence is low, and we found *some* search results (even from untrusted),
    # it's better to be uncertain than confidently wrong.
    if model_confidence < 0.6 and any_search_results_found:
        return "Uncertain"
    
    # If no external evidence found at all (empty fact_checks list)
    if not any_search_results_found:
        if model_confidence < 0.8: # Higher confidence needed if no external checks
            return "Uncertain"
        return model_label # Fallback to model only if confident and no external sources
    
    return model_label # Fallback to model's initial label if no strong external signal


# --- Simplified Fact-Checking Logic (replacing agent workflow) ---

async def perform_agent_fact_check(text: str, lang_code: str):
    """
    Performs a simplified source-based fact-check.
    It directly fetches from APIs and scrapes DuckDuckGo, then uses Gemini to synthesize.
    The source fetching is conditional based on the language.
    Returns a dictionary with the verdict, explanation, and sources.
    """
    print(f"\n--- Initiating Simplified Source-Based Fact-Check for '{text[:50]}...' (Lang: {lang_code}) ---")
    
    claim = text # The input text is the claim for this simplified flow
    fact_checks_formatted = []

    # --- Conditional Source Fetching Logic ---
    if lang_code in ["si", "ta"]:
        # For Sinhala and Tamil, use web scraping (DuckDuckGo)
        print(f"Language is {lang_code}. Using web scraping (DuckDuckGo).")
        search_query = f"{claim} news Sri Lanka" # Added "Sri Lanka" for context
        try:
            scraped_results = await scrape_search_results(search_query, language_code=lang_code)
            fact_checks_formatted.extend(scraped_results)
            print(f"Scraped {len(scraped_results)} results from DuckDuckGo for {lang_code}.")

        except Exception as e:
            print(f"Error during web scraping for {lang_code}: {e}")

    elif lang_code == "en":
        # For English, use Google Fact Check, NewsAPI, and GNews
        print("Language is English. Using Google Fact Check, NewsAPI, and GNews.")
        
        # 1. Google Fact Check API
        google_results = await fetch_google_fact_check(claim)
        if google_results:
            print(f"Google Fact Check API returned {len(google_results)} claims.")
            for fc_item in google_results:
                if "claimReview" in fc_item: # From Google Fact Check API
                    claim_review = fc_item.get("claimReview", [{}])[0]
                    fact_checks_formatted.append({
                        "claim": fc_item.get("text", ""),
                        "review_text": claim_review.get("text", ""),
                        "publisher": claim_review.get("publisher", {}).get("name", ""),
                        "url": claim_review.get("url", "")
                    })

        # 2. NewsAPI & GNews
        newsapi_results = await fetch_newsapi_articles(claim, language_code="en")
        gnews_results = await fetch_gnews_articles(claim, language_code="en")
        
        if newsapi_results or gnews_results:
            print(f"NewsAPI/GNews returned {len(newsapi_results) + len(gnews_results)} articles.")
            fact_checks_formatted.extend(newsapi_results)
            fact_checks_formatted.extend(gnews_results)
        else:
            print("NewsAPI/GNews returned no articles.")
        
        # Optionally, you can still add web scraping for English if you want broader results
        # even with the specific APIs. Uncomment if desired:
        # print(f"Also performing web scraping for: '{claim}' (Language: {lang_code})")
        # try:
        #     scraped_results = await scrape_search_results(f"{claim} news", language_code="en")
        #     fact_checks_formatted.extend(scraped_results)
        # except Exception as e:
        #     print(f"Error during web scraping for EN: {e}")

    else:
        # Fallback for other languages, use web scraping
        print(f"Language is {lang_code} (other). Defaulting to web scraping (DuckDuckGo).")
        search_query = f"{claim} news {lang_code}"
        try:
            scraped_results = await scrape_search_results(search_query, language_code=lang_code)
            fact_checks_formatted.extend(scraped_results)
            print(f"Scraped {len(scraped_results)} results from DuckDuckGo for {lang_code}.")

        except Exception as e:
            print(f"Error during web scraping for {lang_code}: {e}")


    # --- Ask Gemini to synthesize source information and provide a source-based label/confidence ---
    source_synthesis_prompt = f"""
    You are an AI assistant tasked with analyzing information from various sources about a claim.
    Based on the following claim and the provided external search results, synthesize the information to determine if the claim is 'Real', 'Fake', or 'Uncertain'.
    Provide a confidence score between 0.0 and 1.0 (e.g., 0.9 for high confidence) based *only* on the provided sources.
    If sources conflict or are insufficient, state 'Uncertain'.

    Claim in {lang_code}:
    ---
    {claim}
    ---

    Collected External Information:
    """
    if fact_checks_formatted:
        for i, fc in enumerate(fact_checks_formatted[:10]): # Limit to top 10 for prompt length
            source_synthesis_prompt += f"\n--- Source {i+1} ---\n"
            source_synthesis_prompt += f"Publisher: {fc.get('publisher', 'N/A')}\n"
            source_synthesis_prompt += f"Snippet/Review: {fc.get('review_text', 'N/A')[:300]}...\n" # Limit snippet length
            if fc.get('url') and fc.get('url') != "N/A":
                source_synthesis_prompt += f"URL: {fc['url']}\n"
    else:
        source_synthesis_prompt += "\nNo significant external information found. Based on this, the claim is Uncertain."

    source_synthesis_prompt += """
    ---
    Based on the above, provide your assessment.
    Output format: Label: [Real/Fake/Uncertain], Confidence: [0.0-1.0]
    """

    model_label = "Uncertain"
    model_confidence = 0.5
    try:
        llm_response = await gemini_direct_model.generate_content_async(source_synthesis_prompt)
        pred_text = llm_response.text.strip()
        
        label_match = re.search(r"Label:\s*(Real|Fake|Uncertain)", pred_text, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*([0-9.]+)", pred_text)
        
        if label_match:
            model_label = label_match.group(1).capitalize()
        if conf_match:
            model_confidence = float(conf_match.group(1))
        print(f"Source-Based LLM Assessment: Label={model_label}, Confidence={model_confidence:.2f}")

    except Exception as e:
        print(f"Error getting source-based assessment from LLM: {e}")
        # Keep default values if re-assessment fails

    # Make the final verdict using your existing logic, now with source-informed model_label/confidence
    final_verdict = decide_final_verdict(model_label, model_confidence, fact_checks_formatted, detected_lang=lang_code)

    # Format output for the API response
    sources_info = []
    for fc in fact_checks_formatted[:5]: # Limit to top 5 for concise output
        pub = fc.get("publisher", "Unknown")
        review = fc.get("review_text", "No details")
        url = fc.get("url", "No URL")
        sources_info.append(f"- Publisher: {pub}, Review: {review[:100]}..., URL: {url}")
    
    sources_str = "\n".join(sources_info) if sources_info else "No specific external sources found."

    # Return structured data for the FastAPI endpoint
    return {
        "final_verdict": final_verdict,
        "explanation": f"Based on available sources, the verdict is {final_verdict}. AI's Source-Based Label: {model_label} (Confidence: {model_confidence:.2f}).\n\nSupporting External Sources:\n{sources_str}",
        "fact_check_status": "Found" if fact_checks_formatted else "Not found",
        "fact_check_sources": fact_checks_formatted # Now returning the full formatted sources
    }
