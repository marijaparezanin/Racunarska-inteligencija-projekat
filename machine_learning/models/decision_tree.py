import uuid
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_evaluate_best_dt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = [
        {"max_depth": d, "min_samples_split": s}
        for d in [3, 5, 10, None]
        for s in [2, 5, 10]
    ]

    best_model = None
    best_score = -np.inf
    best_params = {}

    for params in param_grid:
        model = DecisionTreeRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    # Final evaluation
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Plot actual vs predicted
    os.makedirs("outputs", exist_ok=True)
    plot_path = f"outputs/dt_actual_vs_pred_{uuid.uuid4().hex}.png"

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual BMI")
    plt.ylabel("Predicted BMI")
    plt.title("Actual vs Predicted BMI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "model": "DecisionTreeRegressor",
        "r2_score": r2,
        "mae": mae,
        "best_params": best_params,
        "actual_vs_pred_path": plot_path
    }
