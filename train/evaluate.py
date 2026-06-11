import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name="Model", is_dl=False):
    """
    Evaluates a model and prints performance metrics:
    Precision, Recall, F1, Confusion Matrix, and ROC-AUC.
    """
    logger.info(f"--- Evaluating {model_name} ---")
    
    if is_dl:
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = y_pred # Fallback
            
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL (0)', 'FAKE (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"\nROC-AUC Score: {auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        
    print("-" * 40)
