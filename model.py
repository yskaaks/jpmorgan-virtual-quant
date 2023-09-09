import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Prepare features and target for Linear Regression
df['Year'] = df['Dates'].dt.year
df['Month'] = df['Dates'].dt.month
X_lr = df[['Year', 'Month']]
y_lr = df['Prices']

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_lr, y_lr)

def estimate_price_lr(input_date):
    input_date = pd.Timestamp(input_date)
    input_year = input_date.year
    input_month = input_date.month
    return lr_model.predict(np.array([[input_year, input_month]]))[0]

def price_contract_lr(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, 
                      max_volume, storage_cost, injection_cost=0, withdrawal_cost=0):
    contract_value = 0
    current_storage = 0
    last_date = None
    all_dates = sorted(set(injection_dates + withdrawal_dates))
    
    for date in all_dates:
        if last_date:
            months_between = (date.year - last_date.year) * 12 + date.month - last_date.month
            contract_value -= months_between * storage_cost * current_storage
        
        if date in injection_dates:
            purchase_price = estimate_price_lr(date)
            inject_volume = min(max_volume - current_storage, injection_rate)
            contract_value -= (purchase_price + injection_cost) * inject_volume
            current_storage += inject_volume
        
        if date in withdrawal_dates:
            selling_price = estimate_price_lr(date)
            withdraw_volume = min(current_storage, withdrawal_rate)
            contract_value += (selling_price - withdrawal_cost) * withdraw_volume
            current_storage -= withdraw_volume
        
        last_date = date
    
    return contract_value

# Sample test
injection_dates = [pd.Timestamp('2022-06-30'), pd.Timestamp('2022-07-31')]
withdrawal_dates = [pd.Timestamp('2022-12-31'), pd.Timestamp('2023-01-31')]
injection_rate = 1e6  # MMBtu
withdrawal_rate = 1e6  # MMBtu
max_volume = 2e6  # MMBtu
storage_cost = 1e5  # $ per month per MMBtu
injection_cost = 1e4  # $ per MMBtu
withdrawal_cost = 1e4  # $ per MMBtu

contract_value_lr = price_contract_lr(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, 
                                      max_volume, storage_cost, injection_cost, withdrawal_cost)
print(f"The value of the contract is: {contract_value_lr}")
