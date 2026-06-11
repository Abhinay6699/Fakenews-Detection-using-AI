# TruthLens - Fake News Detection System

A production-grade NLP pipeline and web service for detecting fake news articles using Machine Learning and Deep Learning.

## Features
- **Comprehensive Preprocessing**: NLTK-based pipeline including URL stripping, lemmatization, and stopword removal.
- **Multiple Models Supported**: Evaluates Logistic Regression, Gradient Boosting, and BiLSTM architectures.
- **Automated Data Loading**: Automatically fetches a public Fake News dataset for seamless end-to-end training.
- **REST API**: Flask-based API offering `/predict` and `/health` endpoints.
- **Polished Web UI**: Dark-themed, responsive vanilla HTML/CSS/JS frontend interface for real-time inference.

## Prerequisites
- Python 3.9+
- Recommended: Virtual Environment

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Models**
   The system must train and export a model artifact before the API can serve predictions.
   ```bash
   python train/train.py
   ```
   *Note: This script will download the dataset to `data/`, preprocess it, extract TF-IDF features (up to 50,000 features), train three models (Logistic Regression, Gradient Boosting, BiLSTM), select the best one based on F1-score, and save it to the `models/` directory.*

3. **Run the API Server**
   ```bash
   python run.py
   ```
   The application will be accessible at `http://localhost:5000`.

## Training Details & Output Calculation

### Training Process
When running `train.py`:
1. **Data Loading**: Loads the dataset (sampling 10,000 records for speed if the dataset is large).
2. **Text Preprocessing**: Cleans the text by removing URLs, lemmatizing words, and removing standard English stopwords.
3. **Feature Extraction**: Uses `TfidfVectorizer` with unigrams and bigrams (`ngram_range=(1,2)`) to extract numerical features for traditional ML models, and a `Tokenizer` + `pad_sequences` for the deep learning model.
4. **Model Training**: 
   - **Logistic Regression**: Trained on TF-IDF features.
   - **Gradient Boosting**: Trained on TF-IDF features.
   - **BiLSTM**: A bidirectional LSTM network trained on padded word sequences.
5. **Evaluation & Selection**: All models are evaluated on a 20% holdout test set using the **F1-score**. The model with the highest F1-score is saved to `models/best_model.pkl` (or `.h5` for BiLSTM) along with its corresponding vectorizer/tokenizer.

### Output Calculation
When predicting whether a news article is fake or real via the API:
1. The incoming text is passed through the same NLTK-based preprocessing pipeline.
2. The saved `TfidfVectorizer` (or `Tokenizer`) transforms the cleaned text into numerical features.
3. The selected best model calculates a probability score for the "Fake" class.
4. If the probability score > 0.5, the text is classified as `FAKE`, otherwise `REAL`.
5. The API returns the assigned label and the confidence (the raw probability score).

## API Documentation

### `GET /health`
Verifies API status and model availability.
**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "LogisticRegression"
}
```

### `POST /predict`
Classifies a block of text.
**Headers**: `Content-Type: application/json`
**Body**:
```json
{
  "text": "Your news article content here..."
}
```
**Response**:
```json
{
  "label": "FAKE",
  "confidence": 0.9421,
  "model_used": "LogisticRegression"
}
```

## Running Tests
Run the test suite using `pytest`:
```bash
pytest tests/
```

## Project Structure
- `app/`: Flask application, UI templates, and prediction logic.
- `train/`: Scripts for data fetching, model training, and evaluation.
- `models/`: Persistent storage for trained `.pkl` and `.h5` model artifacts.
- `tests/`: Pytest suite for unit and integration testing.
