import streamlit as st
import requests
import os
import time

st.title("Machine Learning Model Training")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Reduce sidebar top padding */
        section[data-testid="stSidebar"] .css-1d391kg {
            padding-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
This model is based on the [CDC Diabetes Health Indicators Dataset](https://huggingface.co/datasets/Bena345/cdc-diabetes-health-indicators) available on Hugging Face.  
The source code for this application can be found on the [GitHub repository](https://github.com/marijaparezanin/visualization_ml_predictor).
""")

# Sidebar: Classification Configuration
st.sidebar.header("Classification Model Configuration")
model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "K Nearest Neighbors", "Gradient Boosting"], key="clf_model")
imputer_strategy = st.sidebar.selectbox("Select Imputation Strategy", ["mean", "median"], key="clf_imputer")
train_clf = st.sidebar.button("Train Classification Model")

# Sidebar: Regression Configuration
st.sidebar.markdown("---")
st.sidebar.header("Regression Model Configuration")
reg_model_type = st.sidebar.selectbox("Select Regression Model", ["Linear Regression", "Decision Tree"], key="reg_model")
reg_imputer_strategy = st.sidebar.selectbox("Select Regression Imputation Strategy", ["mean", "median"], key="reg_imputer")
train_reg = st.sidebar.button("Train Regression Model")

# Model label to backend key mapping
model_mapping = {
    "Random Forest": "rf",
    "K Nearest Neighbors": "knn",
    "Linear Regression": "lr",
    "Decision Tree": "dt",
    "Gradient Boosting": "gb"
}

# Shared function for training
def send_training_request(payload, display_label):
    with st.spinner(f"Training {display_label}... Please wait."):
        start_time = time.time()
        try:
            response = requests.post("http://localhost:5000/train", json=payload)
            if response.status_code == 200:
                end_time = time.time()
                training_time = round(end_time - start_time, 2)
                st.success(f"Training {display_label} completed in {training_time} seconds.")
                data = response.json()

                # Handle Classification Results
                if "accuracy" in data:
                    st.metric("Accuracy", f"{data['accuracy']:.4f}")

                    for plot_label, plot_key in [
                        ("Bar Plot", "bar_plot_path"),
                        ("Log Loss Plot", "log_loss_plot_path"),
                        ("Training vs Validation Loss Plot", "training_validation_loss_path")
                    ]:
                        if data.get(plot_key):
                            st.subheader(plot_label)
                            st.image(data[plot_key], use_container_width=True)
                            st.download_button(
                                label=f"Download {plot_label}",
                                data=open(data[plot_key], "rb").read(),
                                file_name=os.path.basename(data[plot_key]),
                                mime="image/png"
                            )

                # Handle Regression Results
                if "r2_score" in data and "mae" in data:
                    st.metric("R² Score", f"{data['r2_score']:.4f}")
                    st.metric("Mean Absolute Error", f"{data['mae']:.4f}")

                    if "best_params" in data:
                        st.subheader("Best Hyperparameters")
                        st.json(data["best_params"])

                    if "actual_vs_pred_path" in data:
                        st.subheader("Actual vs Predicted BMI")
                        st.image(data["actual_vs_pred_path"], use_container_width=True)
                        st.download_button(
                            label="Download Plot",
                            data=open(data["actual_vs_pred_path"], "rb").read(),
                            file_name=os.path.basename(data["actual_vs_pred_path"]),
                            mime="image/png"
                        )

            else:
                st.error(f"Error: {response.json().get('error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {str(e)}")

# Button triggers
if train_clf:
    payload = {
        "model": model_mapping[model_type],
        "dataset": 'Diabetes Indicators',
    }
    label = f"{model_type} with {imputer_strategy} imputation"
    send_training_request(payload, label)

if train_reg:
    payload = {
        "model": model_mapping[reg_model_type],
        "dataset": 'Diabetes Indicators',
    }
    label = f"{reg_model_type} with {reg_imputer_strategy} imputation"
    send_training_request(payload, label)
