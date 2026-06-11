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
from sklearn.metrics import f1_score, classification_report

# Ensure the parent directory is in the path to import app modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.preprocessor import preprocessor
from data_loader import DataLoader
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if we are running inside Docker (training at build time)
# In Docker, skip the slow BiLSTM to keep build time within Render's limits.
# Set ENABLE_BILSTM=1 to re-enable.
ENABLE_BILSTM = os.environ.get("ENABLE_BILSTM", "0") == "1"

if ENABLE_BILSTM:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping


class ModelTrainer:
    def __init__(self):
        base_path = Path(__file__).resolve().parent.parent
        self.models_dir = os.path.join(base_path, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.random_state = 42

    def prepare_data(self, df):
        """Preprocesses text and splits into train/test sets."""
        logger.info("Preprocessing texts...")
        X = df['text'].apply(preprocessor.preprocess)
        y = df['target'].values
        logger.info("Splitting dataset...")
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

    def train_logistic_regression(self, X_train_tfidf, y_train):
        logger.info("Training Logistic Regression...")
        clf = LogisticRegression(random_state=self.random_state, max_iter=1000, C=1.0)
        clf.fit(X_train_tfidf, y_train)
        return clf

    def train_gradient_boosting(self, X_train_tfidf, y_train):
        logger.info("Training Gradient Boosting...")
        clf = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        clf.fit(X_train_tfidf, y_train)
        return clf

    def train_bilstm(self, X_train, y_train, X_test, y_test, max_features=50000, maxlen=500):
        logger.info("Training BiLSTM...")
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq  = tokenizer.texts_to_sequences(X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
        X_test_pad  = pad_sequences(X_test_seq,  maxlen=maxlen)

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
            epochs=5, batch_size=64,
            validation_data=(X_test_pad, y_test),
            callbacks=[es], verbose=1
        )

        with open(os.path.join(self.models_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)

        return model, tokenizer, maxlen

    def run_training_pipeline(self):
        loader = DataLoader()
        df = loader.load_data()

        # Use full dataset for best accuracy (LR + GBT handle it fine)
        logger.info(f"Training on {len(df)} records (full dataset).")

        X_train, X_test, y_train, y_test = self.prepare_data(df)

        logger.info("Extracting TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf  = tfidf_vectorizer.transform(X_test)

        # 1. Logistic Regression (always trained)
        lr_model = self.train_logistic_regression(X_train_tfidf, y_train)
        lr_preds  = lr_model.predict(X_test_tfidf)
        lr_f1     = f1_score(y_test, lr_preds)
        logger.info(f"Logistic Regression F1: {lr_f1:.4f}")
        logger.info(f"\n{classification_report(y_test, lr_preds, target_names=['REAL','FAKE'])}")

        models = [("Logistic_Regression", lr_model, lr_f1, "sklearn")]

        # 2. Gradient Boosting (always trained)
        gb_model = self.train_gradient_boosting(X_train_tfidf, y_train)
        gb_preds  = gb_model.predict(X_test_tfidf)
        gb_f1     = f1_score(y_test, gb_preds)
        logger.info(f"Gradient Boosting F1: {gb_f1:.4f}")
        models.append(("Gradient_Boosting", gb_model, gb_f1, "sklearn"))

        # 3. BiLSTM (optional — disabled in Docker by default)
        if ENABLE_BILSTM:
            bilstm_model, tokenizer, maxlen = self.train_bilstm(X_train, y_train, X_test, y_test)
            X_test_pad   = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)
            lstm_preds_p = bilstm_model.predict(X_test_pad)
            lstm_preds   = (lstm_preds_p > 0.5).astype(int).flatten()
            lstm_f1      = f1_score(y_test, lstm_preds)
            logger.info(f"BiLSTM F1: {lstm_f1:.4f}")
            models.append(("BiLSTM", bilstm_model, lstm_f1, "keras"))
        else:
            logger.info("BiLSTM skipped (ENABLE_BILSTM not set). Using sklearn models only.")

        # Select best model by F1
        best_model_name, best_model, best_f1, best_type = max(models, key=lambda x: x[2])
        logger.info(f"Best model: {best_model_name} — F1: {best_f1:.4f}")

        # Save vectorizer
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        logger.info("Saved tfidf_vectorizer.pkl")

        # Save best model
        if best_type == "sklearn":
            with open(os.path.join(self.models_dir, 'best_model.pkl'), 'wb') as f:
                pickle.dump(best_model, f)
            logger.info("Saved best_model.pkl")
            evaluate_model(best_model, X_test_tfidf, y_test, best_model_name)
        else:
            best_model.save(os.path.join(self.models_dir, 'best_model.h5'))
            logger.info("Saved best_model.h5")
            evaluate_model(best_model, X_test_pad, y_test, best_model_name, is_dl=True)

        logger.info("Training pipeline complete.")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
