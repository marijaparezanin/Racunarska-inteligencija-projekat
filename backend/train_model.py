from machine_learning.data_loaders.data_loader_indicators import load_diabetes_dataset as load_indicators_dataset, preprocess_data as preprocess_indicators_data
from machine_learning.data_loaders.data_loader_prediction import load_diabetes_dataset as load_prediction_dataset, preprocess_data as preprocess_prediction_data
from machine_learning.models.random_forest import train_and_evaluate_best_rf
from machine_learning.models.knn import train_and_evaluate_best_knn
from machine_learning.models.decision_tree import train_and_evaluate_best_dt
from machine_learning.models.gradient_boosting import train_and_evaluate_best_gb

def train_model(model_type, dataset):
    """
    Train a machine learning model based on the specified type and dataset.
    
    Args:
        model_type (str): The type of model to train ('rf', 'knn', 'dt', 'gb', 'lr').
        dataset (str): The dataset to use ('Diabetes Indicators' for indicators, 'Diabetes Prediction' for prediction).
    
    Returns:
        dict: A dictionary containing the trained model and evaluation metrics.
    """
    if model_type not in ['rf', 'knn', 'dt', 'gb', 'lr']:
        return {"error": "Invalid model type. Use 'rf', 'knn', 'dt', 'gb' or 'lr'."}, 400

    if dataset == "Diabetes Indicators":
        df = load_indicators_dataset()
        X, y, _ = preprocess_indicators_data(df)
    elif dataset == "Diabetes Prediction":
        df = load_prediction_dataset()
        X, y, _ = preprocess_prediction_data(df)

    if model_type == "rf":
        result = train_and_evaluate_best_rf(X, y)
    elif model_type == "knn":
        result = train_and_evaluate_best_knn(X, y)
    elif model_type == "dt":
        result = train_and_evaluate_best_dt(X, y)
    elif model_type == "gb":
        result = train_and_evaluate_best_gb(X, y)

    return result