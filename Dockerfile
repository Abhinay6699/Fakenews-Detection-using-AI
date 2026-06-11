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
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Print versions for debugging
RUN python -c "import sklearn, numpy, tensorflow; print(f'sklearn={sklearn.__version__} numpy={numpy.__version__} tf={tensorflow.__version__}')"

# Copy project
COPY . /app/

# Validate models load correctly at build time
RUN python -c "\
import pickle, sys; \
v = pickle.load(open('models/tfidf_vectorizer.pkl','rb')); \
result = v.transform(['test']); \
print(f'Vectorizer OK — vocab size: {len(v.vocabulary_)}, shape: {result.shape}'); \
m = pickle.load(open('models/best_model.pkl','rb')); \
print(f'Model OK — type: {type(m).__name__}')"

# Expose port
EXPOSE 5000

# Command to run the application using Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 run:app
