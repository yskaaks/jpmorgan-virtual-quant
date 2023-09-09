import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, plot_importance

# Load the data from the CSV file
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Visualize the data
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))
sns.lineplot(x='Dates', y='Prices', data=df)
plt.title('Natural Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

df['Year'] = df['Dates'].dt.year
df['Month'] = df['Dates'].dt.month

# Features and target variable
X = df[['Year', 'Month']]
y = df['Prices']

# Fitting the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X, y)

# Function to estimate price
def estimate_price_xgb(input_date):
    input_date = pd.Timestamp(input_date)
    input_year = input_date.year
    input_month = input_date.month
    
    if (input_date >= df['Dates'].min()) and (input_date <= df['Dates'].max()):
        return df.loc[df['Dates'] == input_date, 'Prices'].values[0]
    
    predicted_price = xgb_model.predict(np.array([[input_year, input_month]]))
    
    return predicted_price[0]

# Visualizing feature importances
plot_importance(xgb_model)
plt.title('XGBoost Feature Importances')
plt.show()

# Testing the function with a past date and a future date
past_date = '2021-01-31'
future_date = '2025-01-31'
past_price_xgb = estimate_price_xgb(past_date)
future_price_xgb = estimate_price_xgb(future_date)

print(f"Estimated price for past date {past_date}: {past_price_xgb}")
print(f"Estimated price for future date {future_date}: {future_price_xgb}")
