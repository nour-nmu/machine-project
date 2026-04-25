# Customer Churn Prediction

A machine learning web application that predicts customer churn for a telecom company using Logistic Regression.

## 📋 Project Overview

This project predicts whether a telecom customer will churn (cancel their service) based on their demographic information, services subscribed, and billing details.

### Key Features

- **Interactive Web Interface**: Built with Streamlit for real-time predictions
- **Machine Learning Model**: Logistic Regression classifier
- **Preprocessing Pipeline**: StandardScaler for numerical features + OneHotEncoder for categorical features
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Dataset

- **Source**: Telco Customer Churn dataset (WA*Fn-UseC*-Telco-Customer-Churn.csv)
- **Features**: 21 features including demographics, services, and billing information
- **Target**: Churn (Yes/No)

## 🚀 How to Use

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the App

1. **Enter Customer Data**: Use the sidebar to input customer information:
   - Demographics (gender, senior citizen status, partner, dependents)
   - Services (phone, internet, security, backup, etc.)
   - Billing (contract type, payment method)
   - Account details (tenure, monthly charges, total charges)

2. **View Prediction**: The main panel displays:
   - Churn prediction (Yes/No)
   - Probability score
   - Model performance metrics
   - Feature importance visualization

### Model Files

The app automatically:

- Trains a new model if `log_reg.pkl` and `transformer.pkl` don't exist
- Loads pre-trained model files if available

## 📊 Results Analysis

### Model Performance

The Logistic Regression model achieves the following metrics on the test set:

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | ~75-80% |
| Precision | ~60-70% |
| Recall    | ~50-60% |
| F1-Score  | ~55-65% |
| ROC-AUC   | ~80-85% |

### Key Insights

1. **Tenure**: Longer customer tenure strongly correlates with lower churn
2. **Contract Type**: Month-to-month contracts have higher churn rates than long-term contracts
3. **Internet Service**: Fiber optic customers show different churn patterns than DSL
4. **Payment Method**: Electronic check payments are associated with higher churn
5. **Monthly Charges**: Higher monthly charges correlate with increased churn risk

### Feature Importance

Top predictive features (based on model coefficients):

- Contract type (Month-to-month vs longer terms)
- Tenure
- Internet service type
- Payment method
- Monthly charges

## 📁 Project Structure

```
├── app.py                      # Streamlit web application
├── churn_prediction.ipynb      # Jupyter notebook with model training
├── requirements.txt            # Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── log_reg.pkl                 # Trained model (auto-generated)
└── transformer.pkl             # Fitted preprocessor (auto-generated)
```

## 🔧 Technical Details

- **Framework**: Streamlit
- **ML Library**: scikit-learn
- **Model**: Logistic Regression with `liblinear` solver
- **Preprocessing**:
  - Numerical: StandardScaler
  - Categorical: OneHotEncoder (drop first category)
- **Train/Test Split**: 80/20 with random_state=1

## 📝 License

This project is for educational purposes.
