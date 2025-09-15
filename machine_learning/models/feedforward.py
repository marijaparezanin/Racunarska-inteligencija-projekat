import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score
from machine_learning.models.utils import bar_plot, plot_torch_loss, plot_torch_confusion_matrix, plot_torch_roc
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(FeedForwardNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_and_evaluate_torch_nn(X, y, dataset_name, hidden_dims=None, dropout=0.3, epochs=50, batch_size=64, lr=0.001):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    input_dim = X_train.shape[1]
    hidden_dims = hidden_dims or [128, 64]

    model = FeedForwardNN(input_dim, hidden_dims, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_t).numpy().flatten()
        y_pred = (y_pred_probs >= 0.5).astype(int)

    # Plots
    loss_curve_path = plot_torch_loss(train_losses, f"torch_nn_{dataset_name}")

    bar_plot_path = bar_plot(y_test, y_pred, save_path=f"outputs/torch_nn_{uuid.uuid4().hex}.png")
    roc_curve_path = plot_torch_roc(y_test, y_pred_probs, dataset_name)
    conf_matrix_path = plot_torch_confusion_matrix(y_test, y_pred, dataset_name)

    return {
        "model": "TorchFeedForwardNN",
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "bar_plot_path": bar_plot_path,
        "loss_curve_path": loss_curve_path,
        "roc_curve_path": roc_curve_path,
        "conf_matrix_path": conf_matrix_path
    }
