import uuid
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score
from machine_learning.models.utils import bar_plot, plot_knn_train_test_loss

def train_and_evaluate_best_knn(X, y, neighbor_range=[3, 5, 7, 9], threshold_range=[0.2, 0.3, 0.4, 0.5]):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neighbors = 5
    thresh = 0.2
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= thresh).astype(int)
    recall = recall_score(y_test, y_pred, pos_label=1)


    plot_path = f"outputs/knn_{uuid.uuid4().hex}.png"
    bar_plot_path = bar_plot(y_test, y_pred, save_path=plot_path)
    training_validation_loss_path = plot_knn_train_test_loss(X_train, y_train, X_test, y_test)

    return {
        "model": "KNearestNeighbors",
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "bar_plot_path": bar_plot_path,     
        "training_validation_loss_path": training_validation_loss_path  
    }
