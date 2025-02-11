# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')
    
    # Label encode categorical columns
    df['Geography'] = LabelEncoder().fit_transform(df['Geography'])
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    
    # Define features and target
    X = df.drop(columns=["Exited"])  # Assuming "Exited" is the target
    y = df["Exited"]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler
