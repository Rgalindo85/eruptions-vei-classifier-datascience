import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- MLflow Imports ---
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
# ----------------------

from plotting import confusion_matrix_plot, classification_report_plot 

# --- MLFLOW CONFIG ---
# Set a local tracking URI; this creates the 'mlruns' folder
MLFLOW_TRACKING_URI = "file:./mlruns"  
EXPERIMENT_NAME = "Volcano_VEI_Classification"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# ---------------------

# Define Hyperparameters to be tracked
N_ESTIMATORS = 20
RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parents[1]

def main():

    print("****** Starting classifier training ******")

    # --- MLflow Run Start: All subsequent logs are associated with this run ---
    run_name = "RandomForest_VEI_Classifier"
    with mlflow.start_run(run_name=run_name) as run:
        
        # Log key parameters before running the training
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_state", RANDOM_STATE)

        df_train, df_test = get_data()
        
        X_train, y_train = prepare_data(df_train)
        X_test, y_test = prepare_data(df_test)
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # check if X_train and X_test have the same columns
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[X_train.columns]

        # Log the final set of features used for model training
        mlflow.log_dict({"features": list(X_train.columns)}, "features.json")

        model = train_classifier(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        df_preds = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_proba': y_proba})

        # Evaluation logs metrics and plots as artifacts
        evaluate_model(df_preds)
        
        # Log the final Scikit-learn pipeline model
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="RandomForestVEIClassifier"
        )
        
        print(f"MLflow Run ID: {run.info.run_id}")
        
    print("DONE!!")


def evaluate_model(df_preds):
    y_test = df_preds['y_true']
    y_pred = df_preds['y_pred']
    
    # 1. Calculate Metrics and Log Scalars
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log key scalar metrics
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    # Assuming '1' is the high VEI class
    mlflow.log_metric("test_f1_score_high_vei", report['1']['f1-score'])
    mlflow.log_metric("test_precision_high_vei", report['1']['precision'])
    mlflow.log_metric("test_recall_high_vei", report['1']['recall'])
    
    # Log the full classification report as a dictionary artifact
    mlflow.log_dict(report, "classification_report.json")

    # 2. Log Plots as Artifacts
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create temporary directory for plots (MLflow logs from file paths)
    temp_plot_dir = "temp_mlflow_plots"
    os.makedirs(temp_plot_dir, exist_ok=True)
    
    plots = {
        "confusion_matrix_normalized.png": confusion_matrix_plot(cm_norm, class_names=np.unique(y_test), title='Normalized Confusion Matrix'),
        "confusion_matrix.png": confusion_matrix_plot(cm, class_names=np.unique(y_test), title='Confusion Matrix'),
        "classification_report.png": classification_report_plot(pd.DataFrame.from_dict(report).transpose(), title='Classification Report')
    }

    for filename, plt_obj in plots.items():
        file_path = os.path.join(temp_plot_dir, filename)
        plt_obj.savefig(file_path)
        # Log the file as an artifact under the 'figures' folder
        mlflow.log_artifact(file_path, "figures")
        plt_obj.close()
        os.remove(file_path) # Clean up temporary file

    os.rmdir(temp_plot_dir)


def train_classifier(X_train, y_train):
    """Initializes and trains the pipeline."""
    # NO joblib.dump() needed here. MLflow handles model persistence in main().
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        # Use global parameters
        ('rf', RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)) 
    ])
    model.fit(X_train, y_train)

    return model


# Helper functions (No changes required for MLflow)
def prepare_data(df):
    features = [
        'latitude_x',
        'longitude_x',
        'elevation',
        'start_year',
    ]
    cat = [
        'eruption_category',
        'event_type',
        'major_rock_1',
    ]
    df_class = pd.get_dummies(df[features + cat], drop_first=True)
    X = df_class
    y = df['vei_class_label']
    return X, y

def get_data():
    df_train = pd.read_csv(os.path.join(PROJECT_DIR, "data", "model_input", "train.csv"))
    df_test = pd.read_csv(os.path.join(PROJECT_DIR, "data", "model_input", "test.csv"))
    return df_train, df_test


if __name__ == "__main__":
    main()