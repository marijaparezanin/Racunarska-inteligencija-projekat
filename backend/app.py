from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import time
import json
from backend.train_model import train_model

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])



CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")


@app.route("/train", methods=["POST"])
def train():
    try:
        model_type = request.json.get("model")
        dataset = request.json.get("dataset")

        if model_type not in ["rf", "knn", "lr", "dt", "gb"]:
            return jsonify({"error": "Invalid model type. Use 'rf', 'knn', 'dt', 'gb' or 'lr'."}), 400


        if dataset not in ["Diabetes Indicators", "Diabetes Prediction"]:
            return jsonify({"error": "Invalid dataset. Use 'Diabetes Indicators' or 'Diabetes Prediction'."}), 400


        # Generate cache file name
        safe_dataset = dataset.replace(" ", "_")
        cache_filename = f"{model_type}__{safe_dataset}.json"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True) 

        # Return cached result if it exists
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached_result = json.load(f)
            cached_result["message"] = f"{model_type} model loaded from cache"
            return jsonify(cached_result)

        # Train model if not cached
        start_time = time.time()
        result = train_model(model_type, dataset)
        duration = round(time.time() - start_time, 2)

        # Prepare response
        response = {
            "message": f"{result['model']} model trained successfully",
            "training_time_seconds": duration,
            "accuracy": result["accuracy"],
            "classification_report": result["classification_report"],
            "bar_plot_path": result["bar_plot_path"],
            "training_validation_loss_path": result["training_validation_loss_path"],
            "duration": duration,
        }
        if model_type == "rf":
            response["log_loss_plot_path"] = result.get("log_loss_plot_path")

        # Save to cache
        with open(cache_path, "w") as f:
            json.dump(response, f, indent=2)

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": "An error occurred during training", "details": str(e)}), 500


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

@app.route("/plot/<path:filename>")
def get_plot(filename):
    filename_only = os.path.basename(filename)
    outputs_path = os.path.join(OUTPUTS_DIR, filename_only)
    if os.path.exists(outputs_path):
        return send_from_directory(OUTPUTS_DIR, filename_only, mimetype="image/png")

    plots_path = os.path.join(PLOTS_DIR, filename_only)
    if os.path.exists(plots_path):
        return send_from_directory(PLOTS_DIR, filename_only, mimetype="image/png")

    return jsonify({"error": "Plot not found"}), 404
