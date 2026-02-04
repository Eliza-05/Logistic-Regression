"""
inference.py - SageMaker Inference Handler

This script is used by SageMaker to:
1. Load the model when the endpoint starts
2. Process incoming prediction requests
3. Return formatted predictions

AWS SageMaker automatically calls these functions:
- model_fn(): Load model at endpoint creation
- input_fn(): Process incoming HTTP requests
- predict_fn(): Make predictions
- output_fn(): Format responses
"""

import json
import numpy as np
import os

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    """
    Load model artifacts when SageMaker endpoint starts.
    SageMaker calls this function once during endpoint initialization.
    """
    # Load model components
    w = np.load(os.path.join(model_dir, "weights.npy"))
    b = np.load(os.path.join(model_dir, "bias.npy"))
    norm_params = np.load(
        os.path.join(model_dir, "normalization_params.npy"),
        allow_pickle=True
    ).item()
    model = {
        'weights': w,
        'bias': b,
        'mu': norm_params['mu'],
        'sigma': norm_params['sigma']
    }
    return model

def input_fn(request_body, content_type='application/json'):
    """
    Parse and preprocess incoming prediction requests.
    SageMaker calls this for each prediction request.
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        # Expected feature order (must match training)
        features_order = [
            'Age', 'BP', 'Cholesterol', 'Max HR',
            'ST depression', 'Number of vessels fluro', 'Exercise angina'
        ]
        # Convert dict to array
        X_raw = np.array([data[feat] for feat in features_order])
        return X_raw
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Make prediction using loaded model.
    SageMaker calls this after input_fn() for each request.
    """
    # Extract model components
    w = model['weights']
    b = model['bias']
    mu = model['mu']
    sigma = model['sigma']

    # Normalize input using training statistics
    X_norm = (input_data - mu) / sigma

    # Compute prediction
    z = np.dot(X_norm, w) + b
    probability = sigmoid(z)
    prediction = 1 if probability >= 0.5 else 0

    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'HIGH RISK (Disease)' if prediction == 1 else 'LOW RISK (Healthy)'
    }

def output_fn(prediction, accept='application/json'):
    """
    Format prediction results for HTTP response.
    SageMaker calls this after predict_fn() to format the response.
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")