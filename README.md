# Customer Churn Prediction (Telco Dataset)

## Overview
This project builds a supervised machine learning model to predict customer churn for a telecom company using the publicly available **Telco Customer Churn dataset** (IBM sample). Customer churn prediction is a key task in marketing analytics, allowing companies to identify at-risk customers and design targeted retention strategies.

The project demonstrates the full data science workflow, including:
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Categorical encoding using One-Hot Encoding
- Model building with Logistic Regression and Random Forest
- Model evaluation with accuracy, precision, recall, F1-score, ROC AUC
- Feature importance analysis
- Interpretation of churn drivers

The dataset is **not included in this repository** due to licensing, but is publicly available on Kaggle.

## Dataset
**Source:** *Telco Customer Churn – IBM Sample Dataset (Kaggle)*  
Link: [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn#)

Each row represents a customer and whether they churned (left the company).  
The target variable is:

- **Churn** (Yes/No → encoded to 1/0)

Key features include:
- **Customer account information:** tenure, contract type, payment method
- **Services:** phone service, internet, device protection, streaming TV
- **Charges:** MonthlyCharges, TotalCharges

The dataset contains both **numerical** and **categorical** variables.

## Project Structure
customer-churn-prediction/
│
├── churn_analysis.ipynb      # Main Jupyter notebook
└── README.md                 # Project documentation

## Methods

### 1. **Data Cleaning**
- Converted `TotalCharges` to numeric (handling invalid entries)
- Removed rows with missing values in key fields
- Encoded target variable (`Churn`: Yes=1, No=0)
- Dropped identifier column `customerID`

### 2. **Exploratory Data Analysis (EDA)**
Visual analysis explored:
- Churn distribution (imbalanced dataset)
- Churn by contract type
- Churn vs tenure
- Churn vs monthly charges
- KDE and histogram plots for key features

### 3. **Preprocessing**
Used `ColumnTransformer` to:
- Apply **OneHotEncoder** to categorical features
- Pass numeric features unchanged

### 4. **Modeling**
Two models were trained:
- **Logistic Regression** (baseline, interpretable)
- **Random Forest Classifier** (higher predictive power)

Both models were implemented using scikit-learn pipelines.

### 5. **Evaluation Metrics**
Models were evaluated using:
- Accuracy  
- Precision  
- Recall (important for identifying churners)  
- F1-score  
- ROC AUC  
- Confusion Matrix  
- ROC Curve  

A custom evaluation function was used to generate consistent reports and plots.

## Results

### Logistic Regression
The Logistic Regression model performed well as a baseline, achieving good overall accuracy and strong ROC AUC performance. It captured churn patterns reasonably well while remaining interpretable.

**Performance (Test Set):**
- **Accuracy:** 0.802  
- **Precision:** 0.644  
- **Recall:** 0.575  
- **F1-score:** 0.607  
- **ROC AUC:** 0.837  

**Interpretation:**  
- Logistic Regression shows balanced performance and relatively strong generalization.
- Its recall (0.57) indicates that it identifies around half of the customers who churn.
- Given its interpretability and stable ROC AUC, it is a solid baseline model.

### Random Forest Classifier
The Random Forest model achieved higher training accuracy but slightly lower generalization compared to Logistic Regression.  
This is typical for tree ensembles when the dataset is moderately imbalanced.

**Performance (Test Set):**
- **Accuracy:** 0.790  
- **Precision:** 0.630  
- **Recall:** 0.513  
- **F1-score:** 0.566  
- **ROC AUC:** 0.827  

**Interpretation:**  
- Random Forest showed mild overfitting (train accuracy 0.93 vs test accuracy 0.79).  
- Its recall (0.51) is lower than Logistic Regression, meaning it catches fewer churners.
- However, Random Forest provides **feature importance**, which helps identify business drivers of churn.

### Feature Importance (Random Forest)
The top predictors of churn include:

1. **TotalCharges**  
2. **tenure**  
3. **MonthlyCharges**  
4. **Contract: Month-to-month**  
5. **OnlineSecurity: No**  
6. **PaymentMethod: Electronic check**  
7. **TechSupport: No**  
8. **InternetService: Fiber optic**  
9. **Contract: Two year**  
10. **OnlineBackup: No**

These results align with real-world telecom churn patterns:

- Customers with **short tenure** and **month-to-month contracts** are more likely to churn.  
- Higher charges contribute to increased churn likelihood.  
- Customers lacking online security or tech support services churn more often.  
- Electronic check customers show higher churn risk.

### Summary of Findings
- **Logistic Regression outperformed Random Forest on the test set**, especially in recall and ROC AUC, making it the better model for this dataset.  
- Random Forest provided valuable interpretability via feature importance, highlighting contract type, tenure, and monthly charges as key churn drivers.  
- From a business perspective, churn reduction strategies should focus on:  
  - Customers on **month-to-month contracts**  
  - Customers with **high monthly charges**  
  - Customers lacking **security/support add-ons**  
  - Customers with **short tenure**

Overall, both models perform reasonably well and provide actionable insights for customer retention.

## How to Run
> Note: The dataset is **not included**. Download the CSV from Kaggle and place it in the same folder as the notebook.

1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
