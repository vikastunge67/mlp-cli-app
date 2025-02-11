import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from torch.utils.data import DataLoader, TensorDataset

# ✅ Define MLP Models (as before)
class SLP(nn.Module):  # Single Layer Perceptron
    def __init__(self, input_size, output_size=1):
        super(SLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

class SLMP(nn.Module):  # Single Layer Multi Perceptron
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(SLMP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

class MLMP(nn.Module):  # Multi-Layer Multi Perceptron
    def __init__(self, input_size, hidden1=64, hidden2=32, output_size=1):
        super(MLMP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# ✅ Preprocess Data
def preprocess_data(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors='ignore')  # Drop irrelevant columns
    df['Geography'] = LabelEncoder().fit_transform(df['Geography'])  # Encoding categorical columns
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Encoding categorical columns
    X = df.drop(columns=["Exited"])  # Features (excluding target)
    y = df["Exited"]  # Target column
    return X, y


# ✅ Train Models & Evaluate
def train_models(X, y, model_type='MLMP', epochs=100, batch_size=64, lr=0.001):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create data loaders
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Select model based on user input
    model_classes = {
        "SLP": SLP,
        "SLMP": SLMP,
        "MLMP": MLMP
    }
    model_class = model_classes.get(model_type, MLMP)  # Default to MLMP
    model = model_class(input_size=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Early Stopping variables
    best_loss = float('inf')
    patience = 5  # Early stopping patience
    epochs_without_improvement = 0

    # Training loop with Early Stopping
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Check for early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "mlp_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    # Load the best model
    model.load_state_dict(torch.load("mlp_best.pth"))

    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predictions = (test_outputs > 0.5).float()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # Print evaluation metrics
    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix: \n{cm}")

    print("Best model saved as mlp_best.pth")

# ✅ Cross-validation for model performance
def cross_validate_model(X, y, model_type='MLMP', epochs=100):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Convert to DataLoader
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Create data loaders
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)

        # Train model
        train_models(X, y, model_type=model_type, epochs=epochs)


        # Evaluate on the test fold
        model = MLMP(input_size=X.shape[1])
        model.load_state_dict(torch.load("mlp_best.pth"))
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predictions = (test_outputs > 0.5).float()

        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    print(f"Cross-validation Accuracy: {sum(accuracies) / len(accuracies):.4f}")

# ✅ Main block for direct CSV reading
if __name__ == "__main__":
    # Directly read the CSV file
    df = pd.read_csv(r'Churn_Modelling.csv')  # Use raw string to escape backslashes

    # Preprocess the data
    X, y = preprocess_data(df)

    #model type (can change this)
    model_type = 'MLMP'  # Change to 'SLP' or 'SLMP' if needed
    epochs = 100  # Number of epochs to train
    batch_size = 64
    learning_rate = 0.001

    # Train and evaluate the model
    train_models(X, y, model_type=model_type, epochs=epochs, batch_size=batch_size, lr=learning_rate)

    # Perform cross-validation (optional)
    # cross_validate_model(X, y, model_type=model_type, epochs=epochs)
