import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import pickle
import logging
from pathlib import Path

# Need to load correct model based on what was saved
try:
    import tensorflow as tf
except ImportError:
    tf = None

from app.preprocessor import preprocessor

logger = logging.getLogger(__name__)

class FakeNewsModel:
    """
    Singleton-like wrapper class to load and serve the best trained model.
    Handles both sklearn and Keras models dynamically based on what was saved.
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.is_dl = False
        self.model_name = "Unknown"
        self.load_model()

    def load_model(self):
        base_path = Path(__file__).resolve().parent.parent
        models_dir = os.path.join(base_path, 'models')
        
        # Load vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            try:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                # Validate the vectorizer is actually fitted
                _ = self.vectorizer.transform(["test validation"])
                logger.info(f"TF-IDF vectorizer loaded and validated. Vocab size: {len(self.vectorizer.vocabulary_)}")
            except Exception as e:
                logger.error(f"CRITICAL: TF-IDF vectorizer failed validation: {type(e).__name__}: {e}")
                self.vectorizer = None
        else:
            logger.error(f"CRITICAL: tfidf_vectorizer.pkl not found at {vectorizer_path}")
        
        # Try loading a scikit-learn model first
        sklearn_model_path = os.path.join(models_dir, 'best_model.pkl')
        dl_model_path = os.path.join(models_dir, 'best_model.h5')
        
        if os.path.exists(sklearn_model_path):
            try:
                with open(sklearn_model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_name = type(self.model).__name__
                self.is_dl = False
                logger.info(f"Loaded sklearn model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading sklearn model: {e}")
        elif os.path.exists(dl_model_path) and tf is not None:
            try:
                self.model = tf.keras.models.load_model(dl_model_path)
                self.model_name = "BiLSTM"
                self.is_dl = True
                
                # Load tokenizer for BiLSTM
                tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                    
                logger.info(f"Loaded Keras model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading keras model: {e}")
        else:
            logger.warning("No trained model found. Please run train/train.py first.")

    # Confidence threshold — below this the model reports UNCERTAIN
    UNCERTAIN_THRESHOLD = 0.70

    def predict(self, text: str) -> dict:
        """
        Predicts whether the given text is FAKE or REAL.
        Returns UNCERTAIN when confidence is below the threshold.
        Returns a dictionary with label, confidence, and model_used.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
            
        processed_text = preprocessor.preprocess(text)
        
        if self.is_dl:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = self.tokenizer.texts_to_sequences([processed_text])
            pad_seq = pad_sequences(seq, maxlen=500)
            fake_prob = float(self.model.predict(pad_seq)[0][0])
        else:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer is not loaded. The model artifacts may be corrupted. Please retrain.")
            features = self.vectorizer.transform([processed_text])
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                # probs[0] = P(REAL), probs[1] = P(FAKE)
                fake_prob = float(probs[1])
            else:
                prediction = self.model.predict(features)[0]
                fake_prob = 1.0 if prediction == 1 else 0.0

        # Determine label based on probability and confidence threshold
        confidence = max(fake_prob, 1.0 - fake_prob)

        if confidence < self.UNCERTAIN_THRESHOLD:
            label = "UNCERTAIN"
        elif fake_prob > 0.5:
            label = "FAKE"
        else:
            label = "REAL"
                
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "model_used": self.model_name
        }

# Global instance for the app
model_service = FakeNewsModel()
