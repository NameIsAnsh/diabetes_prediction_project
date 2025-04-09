# utils.py
import joblib
import numpy as np
from pathlib import Path

def predict_diabetes(features_dict):
    """
    Make a prediction using the final model.
    """
    models_dir = Path('models')
    imputer = joblib.load(models_dir / 'imputer.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')
    feature_names = joblib.load(models_dir / 'feature_names.pkl')
    final_model = joblib.load(models_dir / 'final_model.pkl')

    # Create feature array in the correct order
    features = np.array([[features_dict[feature] for feature in feature_names]])
    
    # Apply preprocessing
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    # Make prediction
    prediction = final_model.predict(features_scaled)[0]
    probability = final_model.predict_proba(features_scaled)[0, 1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
    }
