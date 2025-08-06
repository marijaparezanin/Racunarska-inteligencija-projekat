from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS

from backend.train_model import train_model

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])

@app.route("/train", methods=["POST"])
def train():
    try:
        model_type = request.json.get("model")
        dataset = request.json.get("dataset")

        if model_type not in ["rf", "knn", "lr", "dt", "gb"]:
            return jsonify({"error": "Invalid model type. Use 'rf', 'knn', 'dt', 'gb' or 'lr'."}), 400


        if dataset not in ["Diabetes Indicators", "Diabetes Prediction"]:
            return jsonify({"error": "Invalid dataset. Use 'Diabetes Indicators' or 'Diabetes Prediction'."}), 400


        result = train_model(model_type, dataset)

        response ={
            "message": f"{result['model']} model trained successfully",
        }
        if model_type == "rf":
            response["log_loss_plot_path"] = result.get("log_loss_plot_path")

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
