import torch
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

def load_model_and_scalers(model_path='models/best_model.pth'):
    """Load trained model and scalers"""
    from model import BiLSTM_KAN

    # Load checkpoint (handle checkpoints saved as state_dict or wrapped dict)
    ckpt = torch.load(model_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('module') or 'weight' in k for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    # Detect weather input feature size from checkpoint if available
    weather_dim_saved = None
    for k, v in state_dict.items():
        if k.endswith('weather_mlp.0.weight') or k.endswith('.weather_mlp.0.weight'):
            try:
                weather_dim_saved = v.shape[1]
                break
            except Exception:
                pass

    if weather_dim_saved is None:
        weather_dim_saved = 6

    model = BiLSTM_KAN(seq_dim=96, weather_dim=weather_dim_saved)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Attempt non-strict load to allow missing/mismatched keys
        print(f"State dict load error: {e}. Trying non-strict load.")
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load scalers with fallbacks if files missing
    def safe_load(path, name, fallback_shape=None):
        if os.path.exists(path):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    scaler = joblib.load(path)
                    # If unpickling produced an InconsistentVersionWarning, treat as problematic
                    for warn in w:
                        msg = getattr(warn, 'message', warn)
                        if isinstance(msg, InconsistentVersionWarning) or 'InconsistentVersionWarning' in str(msg):
                            print(f"Warning loading scaler '{name}': {msg}. Using fallback scaler instead.")
                            raise RuntimeError('Incompatible scaler pickle version')
                    return scaler
            except Exception as e:
                print(f"Failed to load scaler '{name}' from {path}: {e}. Creating fallback scaler.")
        else:
            print(f"Scaler '{name}' not found at {path}. Creating fallback scaler.")

        # create a dummy MinMaxScaler fitted on zeros with appropriate shape
        scaler = MinMaxScaler()
        if fallback_shape is None:
            scaler.fit(np.zeros((1, 1)))
        else:
            scaler.fit(np.zeros((1, fallback_shape)))
        return scaler

    scaler_y = safe_load('models/scaler_y.pkl', 'scaler_y', fallback_shape=3)
    scaler_X = safe_load('models/scaler_X.pkl', 'scaler_X', fallback_shape=1)
    scaler_weather = safe_load('models/scaler_weather.pkl', 'scaler_weather', fallback_shape=weather_dim_saved)

    return model, scaler_y, scaler_X, scaler_weather

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