import uuid
import os
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from machine_learning.models.utils import bar_plot, plot_training_validation_loss

def train_and_evaluate_best_gb(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 0.05],
        "max_depth": [3, 4]
    }

    gb = GradientBoostingClassifier(random_state=42)
    grid = GridSearchCV(gb, param_grid, scoring="accuracy", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_val)[:, 1]  # Probabilities for class 1

    # Apply custom threshold
    threshold = 0.3
    y_pred = (y_proba > threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=False)
    logloss = log_loss(y_val, best_model.predict_proba(X_val))

    # Generate plots
    bar_plot_path = bar_plot(y_val, y_pred)
    train_val_loss_path = plot_training_validation_loss(best_model, X_train, y_train, X_val, y_val)

    return {
        "model": "GradientBoostingClassifier",
        "accuracy": acc,
        "classification_report": report,
        "bar_plot_path": bar_plot_path,
        "training_validation_loss_path": train_val_loss_path
    }
