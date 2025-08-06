import uuid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from machine_learning.models.utils import bar_plot, plot_log_loss, plot_training_validation_loss
from sklearn.utils.class_weight import compute_class_weight

def train_and_evaluate_best_rf(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights_array = compute_class_weight(
        class_weight='balanced',  # this tells sklearn to compute balanced weights
        classes=np.unique(y_train),
        y=y_train
    )

    # Convert to dictionary {class_label: weight}
    class_weight_dict = dict(zip(np.unique(y_train), class_weights_array))


    # Train a Random Forest with monitoring
    model = RandomForestClassifier(
        class_weight=class_weight_dict,
        max_depth=10,
        min_samples_split=10,
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Final predictions from the fully trained model
    y_pred = model.predict(X_test)

    # Print classification metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_pred))


    plot_path = f"outputs/rf_{uuid.uuid4().hex}.png"
    bar_plot_path = bar_plot(y_test, y_pred, save_path=plot_path)
    log_loss_path = plot_log_loss(model, X_train, y_train, X_test, y_test)
    training_validation_loss_path = plot_training_validation_loss(model, X_train, y_train, X_test, y_test)


    return {
        "model": "RandomForest",
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "log_loss_plot_path": log_loss_path, 
        "bar_plot_path": bar_plot_path,     
        "training_validation_loss_path": training_validation_loss_path  
    }