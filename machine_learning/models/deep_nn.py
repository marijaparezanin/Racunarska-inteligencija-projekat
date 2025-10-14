import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
from machine_learning.models.utils import bar_plot


class DeepOptimizedNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepOptimizedNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate_deep_nn(X, y, dataset_name, epochs=40, batch_size=256, lr=1e-3, patience=5, max_train_samples=60000):
    # Ensure NumPy arrays
    if hasattr(X, "values"): X = X.values
    if hasattr(y, "values"): y = y.values

    # Downsample if dataset is huge (optional)
    if len(X) > max_train_samples:
        idx = np.random.choice(len(X), max_train_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_train_t, y_train_t)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepOptimizedNN(input_dim=X.shape[1], output_dim=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_losses, val_losses = [], []
    best_val_loss, patience_counter = float("inf"), 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"[{epoch+1}/{epochs}] Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(device))
        y_pred = preds.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    loss_plot_path = f"outputs/deepnn_loss_{uuid.uuid4().hex}.png"
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    bar_plot_path = f"outputs/deepnn_bar_{uuid.uuid4().hex}.png"
    bar_plot(y_test, y_pred, save_path=bar_plot_path)

    return {
        "model": "DeepOptimizedNN",
        "accuracy": acc,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "bar_plot_path": bar_plot_path,
        "training_validation_loss_path": loss_plot_path
    }
