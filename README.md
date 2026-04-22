# 🛡️ AI PhishGuard — Phishing Email Detection System

An advanced AI-powered phishing email detection system that uses 
Machine Learning and Natural Language Processing to identify 
phishing emails in real-time.

---

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| XGBoost (Text) | 97.66% | 0.978 |
| DistilBERT (NLP) | 99.33% | 0.9936 |
| XGBoost (URL) | 98.81% | 0.9880 |
| **Final Ensemble** | **99.33%** | **0.9936** |

---

## 🏗️ Project Architecture

---

## 🔬 How It Works

1. **Preprocessing** — Raw email text is cleaned, HTML stripped,
   URLs extracted
2. **Feature Engineering** — TF-IDF (10,000 features) + 
   heuristic features (7) + URL lexical features (15)
3. **Three Models Run Simultaneously:**
   - XGBoost on text features → probability score
   - DistilBERT deep NLP model → probability score
   - XGBoost on URL features → probability score
4. **Ensemble** — Logistic Regression combines all scores
5. **Final Decision** — Score > 0.5 → PHISHING, else SAFE

---

## 📦 Dataset

| Dataset | Samples | Purpose |
|---|---|---|
| Kaggle Phishing Email Dataset | 82,486 emails | Text model training |
| Mendeley URL Dataset | 450,176 URLs | URL model training |

---

## ⚙️ Tech Stack

- **Language:** Python 3.x
- **ML Framework:** Scikit-learn, XGBoost
- **NLP Model:** DistilBERT (HuggingFace Transformers)
- **Deep Learning:** PyTorch
- **Text Processing:** BeautifulSoup, TF-IDF

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/PhishGuard.git
cd PhishGuard
```

### 2. Create virtual environment
```bash
python -m venv phishguard_env
phishguard_env\Scripts\activate  # Windows
source phishguard_env/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run test
```bash
cd src
python test_prediction.py
```

---

## 📈 Sample Predictions



📧 Phishing Email:
Result      : ⚠️ PHISHING
Confidence  : 99.98%
Final Score : 99.98%

📧 Legitimate Email:
Result      : ✅ SAFE
Confidence  : 0.07%
Final Score : 0.07%

---

## 👨‍💻 Developed By

**Sandipan** — Core Module Developer
- Dataset collection and preparation (82K emails + 450K URLs)
- Data preprocessing and cleaning pipeline
- Feature engineering (TF-IDF, heuristic, URL features)
- Training and evaluating XGBoost text model (97.66%)
- Fine-tuning DistilBERT model (99.33%)
- Training XGBoost URL classifier (98.81%)
- Building final ensemble model (99.33%)
- Local deployment and testing pipeline

---

## ⚠️ Limitations

- Performance limited on emails with no extractable text
- Zero-day phishing domains may evade URL analysis
- Encrypted S/MIME emails not supported currently

---

## 📄 License

This project is for educational purposes.