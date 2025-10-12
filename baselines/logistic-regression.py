import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
BATCH_SIZE = 2048
LR = 1e-3
EPOCHS = 10
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

print(f"USING DEVICE: {DEVICE}")

# load data
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

X_train = torch.tensor(train.drop(columns=["Churn"]).values, dtype=torch.float32)
y_train = torch.tensor(train["Churn"].values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(val.drop(columns=["Churn"]).values, dtype=torch.float32)
y_val = torch.tensor(val["Churn"].values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(test.drop(columns=["Churn"]).values, dtype=torch.float32)
y_test = torch.tensor(test["Churn"].values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[1]
print(f"INPUT DIMENSION: {input_dim}")

# simple logistic regression
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # linear layer only

    def forward(self, x):
        return self.linear(x)  # BCEWithLogitsLoss will handle sigmoid

model = LogisticRegressionModel(input_dim).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(model)

# training loop with early stopping
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = np.inf
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += ((torch.sigmoid(preds) > 0.5).float() == yb).sum().item()
        total += xb.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    model.eval()
    with torch.no_grad():
        val_loss, correct, total = 0, 0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
            correct += ((torch.sigmoid(preds) > 0.5).float() == yb).sum().item()
            total += xb.size(0)

        val_loss /= total
        val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        epochs_no_improve = 0
        os.makedirs("logreg-outputs", exist_ok=True)
        torch.save(model.state_dict(), "logreg-outputs/best_model.pt")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"EARLY STOPPING AT EPOCH {epoch}")
        break

# plots
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logreg-outputs/loss_curve.png", dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(8,5))
plt.plot(train_accs, label="Train Acc", linewidth=2)
plt.plot(val_accs, label="Val Acc", linewidth=2, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logreg-outputs/accuracy_curve.png", dpi=300)
plt.show()
plt.close()

# test evaluation
model.load_state_dict(torch.load("logreg-outputs/best_model.pt"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb)
        all_preds.append(torch.sigmoid(preds).cpu().numpy())
        all_labels.append(yb.numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
y_pred = (all_preds > 0.5).astype(int)

acc = accuracy_score(all_labels, y_pred)
prec = precision_score(all_labels, y_pred)
rec = recall_score(all_labels, y_pred)
f1 = f1_score(all_labels, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [acc, prec, rec, f1]
})
metrics_df.to_csv("logreg-outputs/test_metrics.csv", index=False)

print("\nTEST METRICS:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

cm = confusion_matrix(all_labels, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("logreg-outputs/confusion_matrix.png", dpi=300)
plt.show()
plt.close()

print("\nBEST MODEL SAVED")