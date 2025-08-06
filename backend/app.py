from flask import Flask, request, jsonify, send_file
from machine_learning.data_loader import load_diabetes_dataset, preprocess_data
from machine_learning.models.random_forest import train_and_evaluate_best_rf
from machine_learning.models.knn import train_and_evaluate_best_knn
from machine_learning.models.linear_regression import train_and_evaluate_best_lr
from machine_learning.models.decision_tree import train_and_evaluate_best_dt
from machine_learning.models.gradient_boosting import train_and_evaluate_best_gb
import os

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train_model():
    try:
        model_type = request.json.get("model")
        imputer_strategy = request.json.get("imputer_strategy")
        model_category = request.json.get("model_category")
        if imputer_strategy not in ["mean", "median"]:
            return jsonify({"error": "Invalid imputer strategy. Use 'mean' or 'median'."}), 400

        if model_type not in ["rf", "knn", "lr", "dt", "gb"]:
            return jsonify({"error": "Invalid model type. Use 'rf', 'knn', 'dt', 'gb' or 'lr'."}), 400

        if model_category == "classification":
            df = load_diabetes_dataset('train')
            X, y, _ = preprocess_data(df, 'Diabetes_binary', 'nominal', imputer_strategy=imputer_strategy)
        elif model_category == "regression":
            df = load_diabetes_dataset('train')
            X, y, _ = preprocess_data(df, 'BMI', 'numeric', imputer_strategy=imputer_strategy)

        if model_type == "rf":
            result = train_and_evaluate_best_rf(X, y)
        elif model_type == "knn":
            result = train_and_evaluate_best_knn(X, y)
        elif model_type == "lr":
            result = train_and_evaluate_best_lr(X, y)
        elif model_type == "dt":
            result = train_and_evaluate_best_dt(X, y)
        elif model_type == "gb":
            result = train_and_evaluate_best_gb(X, y)

        response ={
            "message": f"{result['model']} model trained successfully",
        }
        if model_type == "rf":
            response["log_loss_plot_path"] = result.get("log_loss_plot_path")
        if model_category == "regression":
            response["r2_score"] = result["r2_score"]
            response["mae"] = result["mae"]
            response["best_params"] = result["best_params"]
            response["actual_vs_pred_path"] = result["actual_vs_pred_path"]
        if model_category == "classification":
            response["accuracy"] = result["accuracy"]
            response["classification_report"] = result["classification_report"]
            response["bar_plot_path"] =  result["bar_plot_path"]
            response["training_validation_loss_path"] = result["training_validation_loss_path"]
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": "An error occurred during training", "details": str(e)}), 500


@app.route("/plot/<filename>")
def get_plot(filename):
    for folder in ["plots", "outputs"]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return send_file(path, mimetype='image/png')
    return jsonify({"error": "Plot not found"}), 404
