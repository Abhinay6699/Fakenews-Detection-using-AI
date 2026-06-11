import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
import pickle
import logging
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Ensure the parent directory is in the path to import app modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.preprocessor import preprocessor
from data_loader import DataLoader
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        base_path = Path(__file__).resolve().parent.parent
        self.models_dir = os.path.join(base_path, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.random_state = 42

    def prepare_data(self, df):
        """Preprocesses text and splits into train/test sets."""
        logger.info("Preprocessing texts...")
        # Apply preprocessing
        # For performance on large datasets, you might want to use multiprocessing
        # or use a subset. We use a subset here for reasonable training times.
        # Reduce size if memory issues occur.
        X = df['text'].apply(preprocessor.preprocess)
        y = df['target'].values
        
        logger.info("Splitting dataset...")
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

    def train_logistic_regression(self, X_train_tfidf, y_train):
        logger.info("Training Logistic Regression...")
        clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        clf.fit(X_train_tfidf, y_train)
        return clf

    def train_gradient_boosting(self, X_train_tfidf, y_train):
        logger.info("Training Gradient Boosting... (this may take a while)")
        # Limit n_estimators for speed, increase for better performance
        clf = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        clf.fit(X_train_tfidf, y_train)
        return clf

    def train_bilstm(self, X_train, y_train, X_test, y_test, max_features=50000, maxlen=500):
        logger.info("Training BiLSTM...")
        
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
        
        model = Sequential([
            Embedding(input_dim=max_features, output_dim=128, input_length=maxlen),
            Bidirectional(LSTM(units=64)),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        model.fit(
            X_train_pad, y_train,
            epochs=5,
            batch_size=64,
            validation_data=(X_test_pad, y_test),
            callbacks=[es],
            verbose=1
        )
        
        # Save tokenizer and model logic
        with open(os.path.join(self.models_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
            
        return model, tokenizer, maxlen

    def run_training_pipeline(self):
        loader = DataLoader()
        df = loader.load_data()
        
        # To speed up training for the BiLSTM and Gradient Boosting in this example
        # we sample the dataframe if it's too large. 
        if len(df) > 10000:
            logger.info("Sampling dataset for faster training demonstration...")
            df = df.sample(n=10000, random_state=self.random_state)
            
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        logger.info("Extracting TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        # 1. Logistic Regression
        lr_model = self.train_logistic_regression(X_train_tfidf, y_train)
        lr_preds = lr_model.predict(X_test_tfidf)
        lr_f1 = f1_score(y_test, lr_preds)
        logger.info(f"Logistic Regression F1: {lr_f1:.4f}")
        
        # 2. Gradient Boosting
        gb_model = self.train_gradient_boosting(X_train_tfidf, y_train)
        gb_preds = gb_model.predict(X_test_tfidf)
        gb_f1 = f1_score(y_test, gb_preds)
        logger.info(f"Gradient Boosting F1: {gb_f1:.4f}")
        
        # 3. BiLSTM
        bilstm_model, tokenizer, maxlen = self.train_bilstm(X_train, y_train, X_test, y_test)
        X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)
        lstm_preds_prob = bilstm_model.predict(X_test_pad)
        lstm_preds = (lstm_preds_prob > 0.5).astype(int).flatten()
        lstm_f1 = f1_score(y_test, lstm_preds)
        logger.info(f"BiLSTM F1: {lstm_f1:.4f}")
        
        # Evaluate Best Model (Assuming Logistic Regression for fast inference via API)
        # We will select the best model based on F1 and save it.
        models = [
            ("Logistic_Regression", lr_model, lr_f1),
            ("Gradient_Boosting", gb_model, gb_f1),
            ("BiLSTM", bilstm_model, lstm_f1)
        ]
        
        best_model_name, best_model, best_f1 = max(models, key=lambda x: x[2])
        logger.info(f"Best model selected: {best_model_name} with F1: {best_f1:.4f}")
        
        # Save the vectorizer and best model (we'll save LR or GB as standard PKL)
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
            
        if best_model_name in ["Logistic_Regression", "Gradient_Boosting"]:
            with open(os.path.join(self.models_dir, 'best_model.pkl'), 'wb') as f:
                pickle.dump(best_model, f)
            # Evaluate using standard script
            evaluate_model(best_model, X_test_tfidf, y_test, "Best sklearn model")
        else:
            bilstm_model.save(os.path.join(self.models_dir, 'best_model.h5'))
            # Evaluate using standard script for deep learning model
            evaluate_model(bilstm_model, X_test_pad, y_test, "Best BiLSTM model", is_dl=True)
            
        logger.info("Training pipeline complete.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
