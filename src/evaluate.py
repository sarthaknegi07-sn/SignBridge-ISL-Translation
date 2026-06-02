import os
import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import ISLDataset
from src.model import ISLModel

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label encoder
    with open(f"{PROCESSED_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    # Load model
    model = ISLModel(
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(f"{MODELS_DIR}/best_model.pth", map_location=device))
    model.eval()

    # Load test data
    test_dataset = ISLDataset(
        f"{PROCESSED_DIR}/X_test.npy",
        f"{PROCESSED_DIR}/y_test.npy"
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/confusion_matrix.png")
    print(f"\nConfusion matrix saved to {MODELS_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()