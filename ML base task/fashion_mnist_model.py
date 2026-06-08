# =============================================================================
# Fashion-MNIST Neural Network - Custom Architecture with Skip Connection
# =============================================================================
# Architecture (from diagram):
#   Input (28x28) → Flatten (784) → Hidden(784→16) → splits into two branches:
#     Branch A: Hidden(16→8) → Hidden(8→8, skip loop) → Skip-Add(8,8→8)
#     Branch B: Hidden(16→12) → Hidden(12→8)
#   Concatenate(8+8=16) → Output(16→num_classes)
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 1. Dataset & DataLoaders
# =============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # normalise to [-1, 1]
])

# Download Fashion-MNIST (train + test splits)
train_full = datasets.FashionMNIST(root="./data", train=True,
                                    download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False,
                                      download=True, transform=transform)

# Split training set → 80% train / 20% validation
val_size   = int(0.2 * len(train_full))
train_size = len(train_full) - val_size
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
NUM_CLASSES = 10

print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}  |  Test: {len(test_dataset)}")

# =============================================================================
# 2. Model Definition
# =============================================================================

class FashionModel(nn.Module):
    """
    Custom dual-branch network with skip connection, matching the diagram.

    Flow:
        x  →  Flatten  →  shared_hidden (784 → 16)
           ┌─────────────────────────┐
           ▼                         ▼
        branch_a1 (16→8)         branch_b1 (16→12)
           ▼                         ▼
        branch_a2 (8→8)  ←loop   branch_b2 (12→8)
           ▼ (skip-add)
        skip_add (a1_out + a2_out)
           ▼
        Concatenate(skip_out, b2_out)  →  Output (16 → 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Shared stem ──────────────────────────────────────────────────────
        self.flatten       = nn.Flatten()
        self.shared_hidden = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU()
        )

        # ── Branch A ─────────────────────────────────────────────────────────
        self.branch_a1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.branch_a2 = nn.Sequential(nn.Linear(8,  8), nn.ReLU())
        # skip-add: element-wise addition of branch_a1 output and branch_a2 output

        # ── Branch B ─────────────────────────────────────────────────────────
        self.branch_b1 = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
        self.branch_b2 = nn.Sequential(nn.Linear(12,  8), nn.ReLU())

        # ── Head ─────────────────────────────────────────────────────────────
        # Concatenate: (None,8) + (None,8) → (None,16)
        self.output_layer = nn.Linear(16, num_classes)

    def forward(self, x):
        # Shared path
        x = self.flatten(x)              # (B, 784)
        x = self.shared_hidden(x)        # (B, 16)

        # Branch A with skip connection
        a1 = self.branch_a1(x)           # (B, 8)
        a2 = self.branch_a2(a1)          # (B, 8)  ← "loop" in diagram
        skip_out = a1 + a2               # (B, 8)  ← SKIP CONNECTION ADD

        # Branch B
        b1 = self.branch_b1(x)           # (B, 12)
        b2 = self.branch_b2(b1)          # (B, 8)

        # Concatenate both branches → (B, 16)
        combined = torch.cat([skip_out, b2], dim=1)

        # Output
        out = self.output_layer(combined) # (B, 10)
        return out


model = FashionModel(num_classes=NUM_CLASSES).to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 3. Loss & Optimiser
# =============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =============================================================================
# 4. Training & Validation Helpers
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """Run one full training epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate on a DataLoader; return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

    return total_loss / total, correct / total

# =============================================================================
# 5. Training Loop
# =============================================================================

EPOCHS = 20

history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   []
}

print("\n" + "="*60)
print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"{epoch:>6} {train_loss:>12.4f} {train_acc*100:>9.2f}% "
          f"{val_loss:>10.4f} {val_acc*100:>8.2f}%")

print("="*60)

# =============================================================================
# 6. Plots
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, EPOCHS + 1)

# Loss plot
ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", markersize=3)
ax1.plot(epochs_range, history["val_loss"],   label="Val Loss",   marker="s", markersize=3)
ax1.set_title("Loss vs Epochs", fontsize=14)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3)

# Accuracy plot
ax2.plot(epochs_range, [a*100 for a in history["train_acc"]], label="Train Acc", marker="o", markersize=3)
ax2.plot(epochs_range, [a*100 for a in history["val_acc"]],   label="Val Acc",   marker="s", markersize=3)
ax2.set_title("Accuracy vs Epochs", fontsize=14)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → training_plots.png")

# =============================================================================
# 7. Save Model Weights with Pickle
# =============================================================================

weights_path = "fashion_mnist_weights.pkl"
with open(weights_path, "wb") as f:
    pickle.dump(model.state_dict(), f)
print(f"Model weights saved → {weights_path}")

# ── How to reload ─────────────────────────────────────────────────────────────
# with open(weights_path, "rb") as f:
#     state_dict = pickle.load(f)
# model.load_state_dict(state_dict)

# =============================================================================
# 8. Generate submission.csv
# =============================================================================

model.eval()
all_preds = []

with torch.no_grad():
    for images, _ in test_loader:
        images  = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())

submission = pd.DataFrame({
    "ImageId":    range(1, len(all_preds) + 1),
    "Label":      all_preds,
    "ClassName":  [CLASS_NAMES[p] for p in all_preds]
})
submission.to_csv("submission.csv", index=False)
print(f"Predictions saved → submission.csv  ({len(submission)} rows)")
print(submission.head())

# =============================================================================
# 9. Final Test Accuracy
# =============================================================================

test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n✅ Final Test Accuracy : {test_acc*100:.2f}%")
print(f"   Final Test Loss     : {test_loss:.4f}")
