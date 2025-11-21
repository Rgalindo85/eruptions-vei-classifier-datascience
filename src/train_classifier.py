import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plotting import confusion_matrix_plot, classification_report_plot



PROJECT_DIR = Path(__file__).resolve().parents[1]

def main():

    print("****** Starting classifier training ******")
    df_train, df_test = get_data()
    
    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # check if X_train and X_test have the same columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]


    model = train_classifier(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    df_preds = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_proba': y_proba})

    evaluate_model(df_preds)
    print("DONE!!")

def evaluate_model(df_preds):
    y_test = df_preds['y_true']
    y_pred = df_preds['y_pred']
    y_proba = df_preds['y_proba']

    cm = confusion_matrix(df_preds['y_true'], df_preds['y_pred'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plot_path = os.path.join(PROJECT_DIR, "reports", "figures")
    os.makedirs(plot_path, exist_ok=True)

    plt = confusion_matrix_plot(cm_norm, class_names=np.unique(y_test), title='Normalized Confusion Matrix')
    plt.savefig(os.path.join(plot_path, "confusion_matrix_normalized.png"))
    plt.close()

    plt = confusion_matrix_plot(cm, class_names=np.unique(y_test), title='Confusion Matrix')
    plt.savefig(os.path.join(plot_path, "confusion_matrix.png"))
    plt.close()
    
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame.from_dict(report).transpose()
    plt = classification_report_plot(report, title='Classification Report')
    plt.savefig(os.path.join(plot_path, "classification_report.png"))
    plt.close()


def train_classifier(X_train, y_train):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)

    # save model
    model_path = os.path.join(PROJECT_DIR, "models")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, "model.joblib"))

    return model


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
        'primary_volcano_type',
        # 'evidence_category',
        # 'tectonic_settings',
        # 'major_rock_1',
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