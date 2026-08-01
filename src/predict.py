import torch
import numpy as np
import joblib

def load_model_and_scalers(model_path='models/best_model.pth'):
    """Load trained model and scalers"""
    from model import BiLSTM_KAN
    
    model = BiLSTM_KAN(seq_dim=96, weather_dim=6)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    scaler_y = joblib.load('models/scaler_y.pkl')
    scaler_X = joblib.load('models/scaler_X.pkl')
    
    return model, scaler_y, scaler_X

def predict_load(model, X_input, weather_input, scaler_y):
    """
    Predict loads using trained model
    X_input: (batch, 7, 96) - 7 days of 15-min load data
    weather_input: (batch, weather_features)
    """
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        weather_tensor = torch.tensor(weather_input, dtype=torch.float32)
        pred_scaled = model(X_tensor, weather_tensor).numpy()
        pred = scaler_y.inverse_transform(pred_scaled)
    return pred