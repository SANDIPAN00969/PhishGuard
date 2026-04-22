
import pickle
import torch
import numpy as np
import re
import html
import math
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ============================================
#   PHISHGUARD PREDICTION PIPELINE
#   Use this file for deployment
# ============================================

MODELS_PATH = "C:/Users/sandi/Downloads/PhishGuard/models/"
BERT_PATH   = "C:/Users/sandi/Downloads/PhishGuard/models/distilbert_final/"   # update this path at deployment

# --- Load All Models ---
def load_models():
    """Load all trained models into memory."""
    import pickle

    # Load XGBoost text model
    with open(MODELS_PATH + "xgb_model.pkl", "rb") as f:
        xgb_text_model = pickle.load(f)

    # Load XGBoost URL model
    with open(MODELS_PATH + "xgb_url_model.pkl", "rb") as f:
        xgb_url_model = pickle.load(f)

    # Load TF-IDF vectorizer
    with open(MODELS_PATH + "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Load final ensemble model
    with open(MODELS_PATH + "final_ensemble_model.pkl", "rb") as f:
        ensemble_model = pickle.load(f)

    # Load DistilBERT
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = DistilBertTokenizer.from_pretrained(BERT_PATH)
    bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH)
    bert_model = bert_model.to(device)
    bert_model.eval()

    return xgb_text_model, xgb_url_model, tfidf_vectorizer, ensemble_model, tokenizer, bert_model, device

# --- Text Cleaning ---
def clean_email_body(text):
    """Clean raw email text for model input."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

# --- URL Extraction ---
def extract_urls(text):
    """Extract URLs from email text."""
    if not isinstance(text, str):
        return []
    url_pattern = r"https?://[^\s<>\"'{}|\\^`\[\]]+"
    return re.findall(url_pattern, text)

# --- Heuristic Features ---
def extract_heuristic_features(text):
    """Extract heuristic features from email text."""
    if not isinstance(text, str) or text.strip() == "":
        return [0, 0, 0, 0, 0, 0, 0]
    urgency_keywords = [
        "urgent", "immediately", "verify", "suspended", "account",
        "password", "click", "confirm", "update", "expires",
        "limited", "act", "security", "alert", "unusual",
        "winner", "prize", "free", "offer", "congratulations"
    ]
    return [
        len(text),
        len(text.split()),
        sum(len(w) for w in text.split()) / len(text.split()) if text.split() else 0,
        text.count("!"),
        text.count("?"),
        len([w for w in text.split() if any(c.isdigit() for c in w)]),
        sum(1 for kw in urgency_keywords if kw in text.lower())
    ]

# --- URL Features ---
def extract_url_features(url):
    """Extract lexical features from a URL string."""
    if not isinstance(url, str) or url.strip() == "":
        return [0] * 15
    try:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ""
        path     = parsed.path     or ""
    except ValueError:
        return [0] * 15
    suspicious_keywords = [
        "login", "verify", "secure", "account", "update",
        "confirm", "banking", "password", "signin", "paypal"
    ]
    prob        = [url.count(c) / len(url) for c in set(url)]
    url_entropy = -sum(p * math.log2(p) for p in prob)
    return [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("/"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        1 if re.match(r"(\d{1,3}\.){3}\d{1,3}", hostname) else 0,
        url_entropy,
        max(0, len(hostname.split(".")) - 2),
        1 if url.startswith("https") else 0,
        len(path),
        1 if any(kw in url.lower() for kw in suspicious_keywords) else 0,
        sum(c.isdigit() for c in url),
        len(hostname)
    ]

# --- BERT Probability ---
def get_bert_probability(text, tokenizer, bert_model, device):
    """Get DistilBERT phishing probability for email text."""
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device)
        )
    prob = torch.softmax(outputs.logits, dim=1)[:, 1].item()
    return prob

# --- MAIN PREDICTION FUNCTION ---
def predict_email(raw_email_text, models=None):
    """
    Main prediction function.
    Input  : raw email text (string)
    Output : (result, confidence, details)
    """
    # Load models if not provided
    if models is None:
        xgb_text_model, xgb_url_model, tfidf_vectorizer, \
        ensemble_model, tokenizer, bert_model, device = load_models()
    else:
        xgb_text_model, xgb_url_model, tfidf_vectorizer, \
        ensemble_model, tokenizer, bert_model, device = models

    import scipy.sparse as sp

    # Step 1: Clean text
    cleaned_text = clean_email_body(raw_email_text)

    # Step 2: Extract features
    tfidf_features     = tfidf_vectorizer.transform([cleaned_text])
    heuristic_features = np.array([extract_heuristic_features(cleaned_text)])
    combined_features  = sp.hstack([tfidf_features, sp.csr_matrix(heuristic_features)])

    # Step 3: Get model probabilities
    xgb_text_prob = xgb_text_model.predict_proba(combined_features)[0][1]
    bert_prob     = get_bert_probability(cleaned_text, tokenizer, bert_model, device)

    # Step 4: Get ensemble score
    ensemble_input = np.array([[xgb_text_prob, bert_prob]])
    final_prob     = ensemble_model.predict_proba(ensemble_input)[0][1]

    # Step 5: Check for URLs and get URL score
    urls = extract_urls(raw_email_text)
    url_score = None
    if urls:
        url_features = np.array([extract_url_features(urls[0])])
        url_score    = xgb_url_model.predict_proba(url_features)[0][1]
        # Combine text ensemble score with URL score
        final_prob = (final_prob * 0.7) + (url_score * 0.3)

    # Step 6: Final decision
    result     = "PHISHING" if final_prob > 0.5 else "SAFE"
    confidence = round(final_prob * 100, 2)

    return {
        "result"          : result,
        "confidence"      : f"{confidence}%",
        "xgb_text_score"  : round(xgb_text_prob * 100, 2),
        "bert_score"      : round(bert_prob * 100, 2),
        "url_score" : f"{round(url_score * 100, 2)}%" if url_score else "No URL found",
        "final_score"     : round(final_prob * 100, 2)
    }
