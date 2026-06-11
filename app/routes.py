from flask import Blueprint, request, jsonify, render_template
from app.model import model_service
import logging

logger = logging.getLogger(__name__)

# Define blueprint for routing
main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    """Renders the frontend UI."""
    return render_template('index.html')

@main.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify API and model status."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_name": model_service.model_name
    })

@main.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON payload with 'text' to classify.
    Returns JSON with prediction label, confidence, and model name.
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in JSON payload."}), 400
            
        text = data['text']
        
        if not text.strip():
            return jsonify({"error": "Text field cannot be empty."}), 400
            
        result = model_service.predict(text)
        return jsonify(result), 200
        
    except ValueError as ve:
        logger.error(f"Prediction error (Value): {ve}")
        return jsonify({"error": str(ve)}), 503 # Service Unavailable (model not loaded)
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500
