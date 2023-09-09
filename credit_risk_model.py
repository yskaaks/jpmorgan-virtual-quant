import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the sample loan book data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Features and target variable
X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
        'income', 'years_employed', 'fico_score']]
y = df['default']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train the XGBoost model
xgb_model = XGBClassifier(objective='binary:logistic')
xgb_model.fit(X_train, y_train)

def calculate_expected_loss(method, credit_lines, loan_amt, total_debt, income, years_employed, fico_score, recovery_rate=0.1):
    # Standardize the input features
    input_data = scaler.transform([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score]])
    
    if method == 'lr':
        pd = lr_model.predict_proba(input_data)[:, 1][0]
    elif method == 'xgb':
        pd = xgb_model.predict_proba(input_data)[:, 1][0]
    else:
        return "Invalid method"
    
    # Calculate Expected Loss
    expected_loss = loan_amt * (1 - recovery_rate) * pd
    
    return expected_loss

# Example usage
credit_lines = 0
loan_amt = 5221.545193
total_debt = 3915.471226
income = 78039.38546
years_employed = 5
fico_score = 605

expected_loss_lr = calculate_expected_loss('lr', credit_lines, loan_amt, total_debt, income, years_employed, fico_score)
expected_loss_xgb = calculate_expected_loss('xgb', credit_lines, loan_amt, total_debt, income, years_employed, fico_score)

print(f"The expected loss using Logistic Regression is: ${expected_loss_lr}")
print(f"The expected loss using XGBoost is: ${expected_loss_xgb}")
