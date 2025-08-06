# Visualization ML Predictor


## Dataset details and prediction goal

The dataset used is of [diabetes health indicators](https://huggingface.co/datasets/Bena345/cdc-diabetes-health-indicators/). We intend to use classification to predict the diabetes label, however we left the option open when processing data to say which label we'd like to be used. This way we can later choose to implement regression models for calculating let's say BMI. 

Imputation included mapping values like: Age, GenHlth and Diabetes_binary from their text representations (e.g. health GenHlth: poor, fair, good etc) to numerical. We dropped the target column along with the id column since it wasn't relevant to the prediction.

## File Breakdown
- **data_loader.py**  
  Contains two main functions:  
  - `load_diabetes_dataset` — Fetches the dataset.  
  - `preprocess_data` — Handles data imputation and accepts a user-provided label, as long as the label’s data type is specified. With this we were able to have **regression models use this method to prepare data to predict BMI** and **classification models use this to predict Diabetes diagnosis**.
 
- **models/knn.py**
  - `train_and_evaluate_best_knn` - Best model calculation has already been carried out, so the hyperparams are already set. This function also calls bar_plot and plot_knn_train_val_test_loss and returns the data ready for backend.
- **models/random_forest.py** - Same structure as knn.
- **models/svm.py** - SVM algorithm, but was never used because of poor and slow performance.
- **models/gradient_boosting.py** - A point of interest of this implemention was the use of SMOTE to see how to balance having the 1s in the database be in the minority.
- **models/utils.py**
  - `bar_plot` - creates and return the path to the bar plot.
  - `plot_training_validation_loss` - creates and returns the plot for random forest.
  - `plot_knn_train_test_loss` - creates and returns the plot for knn. 

Additionally added regression models:
- **models/linear_regression.py**
- **models/decision_trees.py**



## Running the App with Docker

This project contains:
- A **Flask backend** (`run.py`) for machine learning model APIs.
- A **Streamlit frontend** (`streamlit_app.py`) for interactive visualization and model selection.

### Prerequisites

Make sure you have:
- [Docker](https://www.docker.com/get-started) installed
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+)

---

### Build and Run the App

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Build and start both services**:

   ```bash
   docker compose up --build
   ```

   This will:
   - Build the Docker image
   - Start the Flask backend on **http://localhost:5000**
   - Start the Streamlit frontend on **http://localhost:8501**

3. **Access the App**:

   - Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

### Stopping the App

To stop the running containers, press `Ctrl+C` in your terminal and then run:
```bash
docker compose down
```
---


