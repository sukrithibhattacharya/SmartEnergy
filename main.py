import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ensure local modules can be imported when running this file directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from preprocessing import prepare_data
from model import BiLSTM_KAN
from train import train_model

def main():
    # 1. Prepare data
    print("📊 Preparing data...")
    load_file = '/content/load_data.csv'  # Update path as needed

    (X_train, X_test,
     weather_train, weather_test,
     y_train, y_test,
     scaler_y, scaler_X, scaler_weather) = prepare_data(load_file)

    print(f"✅ Data ready!")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # 2. Create models directory
    os.makedirs('models', exist_ok=True)

    # 3. Train model
    print("🚀 Training model...")
    model, losses = train_model(
        X_train, y_train, weather_train,
        epochs=300,
        lr=0.0003,
        batch_size=32,
        save_dir='models'
    )

    # 4. Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('models/training_loss.png')
    plt.show()
    print("✅ Training loss plot saved!")

    # 5. Save scalers
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_weather, 'models/scaler_weather.pkl')
    print("✅ Scalers saved!")

    # 6. Evaluate on test set
    print("📊 Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        weather_test_tensor = torch.tensor(weather_test, dtype=torch.float32)
        y_pred_scaled = model(X_test_tensor, weather_test_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)

    # 7. Calculate metrics
    targets = ['Average Load', 'Maximum Load', 'Minimum Load']
    results = {}

    for i, name in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100

        results[name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # 8. Plot predictions
    plt.figure(figsize=(15, 10))

    for i, name in enumerate(targets):
        plt.subplot(3, 1, i+1)
        plt.plot(y_true[:50, i], label='Actual', color='blue')
        plt.plot(y_pred[:50, i], label='Predicted', color='red', linestyle='--')
        plt.title(f'{name} - Prediction vs Actual')
        plt.xlabel('Sample')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/predictions.png')
    plt.show()
    print("✅ Predictions plot saved!")

    # 9. Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/results.csv')
    print("✅ Results saved!")

    print("\n🎯 SmartEnergy training complete!")
    print("   Model saved in: models/best_model.pth")
    print("   Results saved in: models/")

if __name__ == "__main__":
    main()