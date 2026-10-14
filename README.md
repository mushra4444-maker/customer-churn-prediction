# Customer Churn Prediction using Python and Scikit-learn

A beginner-friendly machine-learning project that estimates whether a customer may stop using a service. It includes a synthetic customer dataset, a training/evaluation script, and a Streamlit web interface.

> **Dataset note:** `data/customer_churn.csv` is synthetic demo data created for learning. Results are not evidence of real-world model performance. Replace it with a properly licensed, real dataset before making business decisions.

## Features
- Data loading and feature/target separation with pandas
- Missing-value handling and feature preprocessing with scikit-learn pipelines
- One-hot encoding of categorical columns and scaling of numeric columns
- Random Forest classification
- Train/test split, accuracy, ROC-AUC, confusion matrix and classification report
- Saved model with joblib
- Interactive prediction app built with Streamlit

## 1. Open in VS Code
Extract the ZIP file and open the `customer_churn_prediction` folder in VS Code.

## 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, you can run the commands below using `python -m pip` without activating the environment, or use Command Prompt activation: `.venv\Scripts\activate.bat`.

## 3. Install packages
```powershell
python -m pip install -r requirements.txt
```

## 4. Train the model
```powershell
python train_model.py
```
This saves `models/churn_model.joblib` and `models/metrics.json`.

## 5. Start the web app
```powershell
python -m streamlit run app.py
```
Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## Dataset columns
- `customer_id`: demo identifier (excluded from model input)
- `tenure_months`, `phone_service`, `internet_service`, `tech_support`
- `contract_type`, `paperless_billing`, `payment_method`
- `monthly_charges`, `total_charges`
- `churn`: target label (`Yes` or `No`)

## GitHub upload
Create a new GitHub repository named `customer-churn-prediction`, then from this folder run:
```powershell
git init
git add .
git commit -m "Add customer churn prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/customer-churn-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. Do not upload API keys, passwords, or private customer data.

## Suggested resume description
**Customer Churn Prediction | Python, Pandas, Scikit-learn, Streamlit**
- Built a machine-learning classification pipeline to estimate customer churn risk from service and account features.
- Applied preprocessing, one-hot encoding, train/test splitting, and Random Forest classification.
- Evaluated the model with accuracy, ROC-AUC, confusion matrix, and classification report; deployed an interactive Streamlit demo.
