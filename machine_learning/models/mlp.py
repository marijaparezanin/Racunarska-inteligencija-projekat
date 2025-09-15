import uuid
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score
from machine_learning.models.utils import *
from machine_learning.consts import (
    MLP_INDICATORS_HIDDEN, MLP_INDICATORS_ALPHA, MLP_INDICATORS_LR,
    MLP_PREDICTION_HIDDEN, MLP_PREDICTION_ALPHA, MLP_PREDICTION_LR
)

def train_and_evaluate_best_mlp(X, y, dataset_name):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Parametri po datasetu
    if dataset_name == "INDICATORS":
        hidden_layer_sizes = MLP_INDICATORS_HIDDEN or (128, 64)   # default složenija mreža
        alpha = MLP_INDICATORS_ALPHA or 0.0005
        learning_rate_init = MLP_INDICATORS_LR or 0.001
    else:
        hidden_layer_sizes = MLP_PREDICTION_HIDDEN or (64, 32)
        alpha = MLP_PREDICTION_ALPHA or 0.0001
        learning_rate_init = MLP_PREDICTION_LR or 0.001

    # Definicija složenije mreže
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42
    )

    # Treniranje
    model.fit(X_train, y_train)

    # Predikcija
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, pos_label=1)

    # Grafikoni
    plot_path = f"outputs/mlp_{uuid.uuid4().hex}.png"
    bar_plot_path = bar_plot(y_test, y_pred, save_path=plot_path)
    training_validation_loss_path = plot_mlp_train_val_loss(model)

    

    y_proba = model.predict_proba(X_test)[:, 1]  # za ROC

    loss_curve_path = plot_mlp_loss(model, dataset_name)
    roc_curve_path = plot_mlp_roc(y_test, y_proba, dataset_name)
    conf_matrix_path = plot_mlp_confusion_matrix(y_test, y_pred, dataset_name)


    return {
        "model": "MLPClassifier",
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "bar_plot_path": bar_plot_path,
        "training_validation_loss_path": training_validation_loss_path,
        "loss_curve_path": loss_curve_path,
        "roc_curve_path": roc_curve_path,
        "conf_matrix_path": conf_matrix_path
    }
