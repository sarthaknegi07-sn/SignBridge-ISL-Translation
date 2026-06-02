import os
import sys
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import ISLDataset
from src.model import ISLModel

# Config
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10  # early stopping

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = ISLDataset(
        f"{PROCESSED_DIR}/X_train.npy",
        f"{PROCESSED_DIR}/y_train.npy"
    )
    val_dataset = ISLDataset(
        f"{PROCESSED_DIR}/X_val.npy",
        f"{PROCESSED_DIR}/y_val.npy"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load label encoder to get num classes
    with open(f"{PROCESSED_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Model
    model = ISLModel(
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total * 100

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pth")
            print(f"  → Best model saved! Acc: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), f"{MODELS_DIR}/final_model.pth")

    # Save training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig(f"{MODELS_DIR}/training_curves.png")
    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()