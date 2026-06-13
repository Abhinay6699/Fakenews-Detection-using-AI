# TruthLens — Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP_Pipeline-154360?style=for-the-badge)

A production-grade NLP system that classifies news articles as FAKE or REAL using a multi-model pipeline — from classical TF-IDF baselines to BiLSTM deep learning — served via Flask REST API with a polished dark-themed web interface.

---

## What It Does

Submit a headline or article text. Get back a verdict (FAKE/REAL) with a confidence score in real time.

Under the hood: the system trains three models, selects the best by F1-score, and serves it via a REST endpoint. The entire pipeline — data loading, preprocessing, training, evaluation, and inference — is automated end-to-end.

---

## Architecture

```
Raw Text
   │
   ▼
NLTK Pipeline (URL strip → lowercase → lemmatize → stop word removal)
   │
   ├──► TF-IDF Vectorizer (50k features, unigrams + bigrams)
   │         │
   │         ├──► Logistic Regression
   │         └──► Gradient Boosting
   │
   └──► Keras Tokenizer + pad_sequences
             │
             └──► BiLSTM (bidirectional LSTM)
                      │
                      ▼
              Best Model (by F1) → Flask API → Web UI
```

---

## Model Results

| Model | F1-Score | Notes |
|---|---|---|
| Logistic Regression | Baseline | Fast, interpretable |
| Gradient Boosting | Competitive | Ensemble on TF-IDF |
| BiLSTM | Best (typically) | Deep sequential model |

Best model is automatically selected and saved to models/best_model.pkl or .h5.

---

## Prediction Logic

1. Text is cleaned via the same NLTK preprocessing pipeline used at training time
2. The saved vectorizer/tokenizer transforms text to numerical features
3. Model outputs a probability for the FAKE class
4. If probability > 0.5 → FAKE, else → REAL
5. API returns label + confidence score

---

## API

GET /health
Response: { "status": "ok", "model_loaded": true, "model_name": "LogisticRegression" }

POST /predict
Request body: { "text": "Your article text here..." }
Response: { "label": "FAKE", "confidence": 0.9421, "model_used": "LogisticRegression" }

---

## Quickstart

pip install -r requirements.txt
python train/train.py
python run.py
Server runs at http://localhost:5000

Docker:
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector

Deploy to Render: Repository includes render.yaml for one-click deployment. Connect GitHub repo, Render detects config automatically.

---

## Project Structure

Fakenews-Detection-using-AI/
├── app/          # Flask application, UI templates, prediction logic
├── train/        # Data fetching, model training, evaluation scripts
├── models/       # Saved .pkl and .h5 model artifacts
├── tests/        # pytest suite (unit + integration)
├── data/         # Dataset storage
├── Dockerfile
├── render.yaml
└── requirements.txt

---

## Run Tests

pytest tests/

---

## Tech Stack

- NLP: NLTK, SpaCy, TF-IDF (sklearn), Keras Tokenizer
- Models: Logistic Regression, Gradient Boosting, BiLSTM (TensorFlow/Keras)
- API: Flask
- Frontend: Vanilla HTML/CSS/JS (dark theme, responsive)
- Deployment: Docker, Render