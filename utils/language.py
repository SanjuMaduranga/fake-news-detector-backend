import langid

def detect_language(text: str) -> str:
    lang, _ = langid.classify(text)
    if any('\u0D80' <= ch <= '\u0DFF' for ch in text):
        return "si"
    elif any('\u0B80' <= ch <= '\u0BFF' for ch in text):
        return "ta"
    return lang

def map_language_code(code: str):
    if code.startswith("si"):
        return "si"
    elif code.startswith("ta"):
        return "ta"
    elif code.startswith("en"):
        return "en"
    return None
