"""Train and evaluate a customer churn classification model."""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"].map({"No": 0, "Yes": 1})
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = X.select_dtypes(exclude="object").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, target_names=["No churn", "Churn"], output_dict=True),
    }
    joblib.dump(model, MODEL_DIR / "churn_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model trained and saved to models/churn_model.joblib")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["No churn", "Churn"]))

if __name__ == "__main__":
    main()
