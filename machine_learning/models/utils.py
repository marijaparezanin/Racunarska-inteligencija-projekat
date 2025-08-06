from sklearn.metrics import log_loss
import numpy as np
import os
import uuid
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
matplotlib.use('Agg')  # Set the backend to 'Agg', which is for file generation, not GUI.
import matplotlib.pyplot as plt

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)  # Ensure plot directory exists

def bar_plot(y_test, y_pred, save_path=None):
    actual_1 = sum(y_test == 1)
    correct_1 = sum((y_test == 1) & (y_pred == 1))
    actual_0 = sum(y_test == 0)
    correct_0 = sum((y_test == 0) & (y_pred == 0))

    categories = ['Class 0', 'Class 1']
    totals = [actual_0, actual_1]
    corrects = [correct_0, correct_1]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, totals, width, label='Actual Count', color='lightgray')
    plt.bar(x + width / 2, corrects, width, label='Correctly Predicted', color='green')
    plt.xticks(x, categories)
    plt.ylabel('Count')
    plt.title('Actual vs Correct Predictions by Class')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()


    os.makedirs("outputs", exist_ok=True) 
    if save_path is None:
        save_path = os.path.join(PLOT_DIR, f"bar_plot_{uuid.uuid4().hex}.png")

    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_log_loss(model, X_train, y_train, X_test, y_test):
    log_losses = []

    for n_trees in range(10, 20, 3):
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_proba)
        log_losses.append(loss)

    # Plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(10, 20, 3), log_losses, marker='o', color='blue')
    plt.title("Cross-Entropy Loss vs. Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Log Loss (Cross-Entropy)")
    plt.grid(True)
    plt.tight_layout()


    os.makedirs("outputs", exist_ok=True) 
    file_path = os.path.join(PLOT_DIR, f"log_loss_{uuid.uuid4().hex}.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_training_validation_loss(model, X_train, y_train, X_test, y_test):
    training_losses = []
    validation_losses = []

    for n_trees in range(10, 51, 10):  # Example for the number of trees
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)
        
        # Calculate training loss
        y_train_proba = model.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_proba)
        training_losses.append(train_loss)

        # Calculate validation loss
        y_test_proba = model.predict_proba(X_test)
        val_loss = log_loss(y_test, y_test_proba)
        validation_losses.append(val_loss)

    # Plot the training and validation loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(10, 51, 10), training_losses, marker='o', label='Training Loss', color='blue')
    plt.plot(range(10, 51, 10), validation_losses, marker='o', label='Validation Loss', color='red')
    plt.title("Training vs Validation Loss vs. Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs("outputs", exist_ok=True)
    file_path = os.path.join(PLOT_DIR, f"training_validation_loss_{uuid.uuid4().hex}.png")
    plt.savefig(file_path)
    plt.close()

    return file_path

def plot_knn_train_test_loss(X_train, y_train, X_test, y_test):
    training_losses = []
    test_losses = []

    neighbor_range = range(1, 11, 2)  # Try odd values of k from 1 to 19

    for k in neighbor_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)

        train_loss = log_loss(y_train, y_train_proba)
        test_loss = log_loss(y_test, y_test_proba)

        training_losses.append(train_loss)
        test_losses.append(test_loss)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(neighbor_range, training_losses, label="Training Loss", marker='o', color='blue')
    plt.plot(neighbor_range, test_losses, label="Test Loss", marker='o', color='green')
    plt.title("Train vs Test Log Loss for KNN")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    file_path = os.path.join(PLOT_DIR, f"knn_train_val_test_loss_{uuid.uuid4().hex}.png")
    plt.savefig(file_path)
    plt.close()

    return file_path