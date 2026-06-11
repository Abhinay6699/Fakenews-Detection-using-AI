# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download NLTK data required by the preprocessor
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy project source (models dir will be populated by training below)
COPY . /app/

# Train the model at build time so pkl files are always compatible
# with the exact library versions installed above
RUN python train/train.py

# Verify models were created successfully
RUN python -c "\
import pickle; \
v = pickle.load(open('models/tfidf_vectorizer.pkl','rb')); \
result = v.transform(['test']); \
print(f'[BUILD OK] Vectorizer vocab size: {len(v.vocabulary_)}'); \
m = pickle.load(open('models/best_model.pkl','rb')); \
print(f'[BUILD OK] Model type: {type(m).__name__}')"

# Expose port
EXPOSE 5000

# Run with Gunicorn — 1 worker to keep memory within free tier limits
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 run:app
