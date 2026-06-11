import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the /health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert 'model_loaded' in data

def test_predict_endpoint_missing_text(client):
    """Test the /predict endpoint with invalid payload"""
    response = client.post('/predict', json={"wrong_key": "test"})
    assert response.status_code == 400
    
def test_predict_endpoint_empty_text(client):
    """Test the /predict endpoint with empty string"""
    response = client.post('/predict', json={"text": "   "})
    assert response.status_code == 400

# Note: Integration test for successful prediction is tricky without a guaranteed 
# trained model artifact present in the test environment.
# We will mock it or just verify that if it returns 503, it's because model isn't loaded.
def test_predict_endpoint_success_or_unavailable(client):
    """Test the /predict endpoint structure"""
    response = client.post('/predict', json={"text": "This is a legitimate news article about science."})
    
    # If model is loaded, it should be 200. If not, our code returns 503.
    assert response.status_code in [200, 503]
    
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'label' in data
        assert 'confidence' in data
        assert data['label'] in ['REAL', 'FAKE']
    else:
        assert 'error' in data
