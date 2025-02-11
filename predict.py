import torch
import pandas as pd
import pickle

# ✅ Load Model & Scaler
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval()
    return model

def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

# ✅ Predict Function
def predict(csv_file, model_path="data/mlp_best.pkl", scaler_path="data/scaler.pkl"):
    df = pd.read_csv(csv_file)

    # Drop unnecessary columns
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

    # Load scaler
    scaler = load_scaler(scaler_path)

    # Encode categorical variables
    df["Geography"] = df["Geography"].astype("category").cat.codes
    df["Gender"] = df["Gender"].astype("category").cat.codes

    # Scale numeric features
    X_scaled = scaler.transform(df.drop(columns=["Exited"], errors="ignore"))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Load model & make predictions
    model = load_model(model_path)
    predictions = model(X_tensor).detach().numpy()
    predictions_binary = (predictions > 0.5).astype(int)

    df["Prediction"] = predictions_binary
    df.to_csv("data/predictions.csv", index=False)
    print("Predictions saved to data/predictions.csv")
