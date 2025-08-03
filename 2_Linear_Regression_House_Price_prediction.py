# Implement simple linear regression to predict house prices based on
features such as area and number of rooms.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('HousingPrice (week 2).csv')

# Calculate area as (longitude * latitude)
df['area'] = abs(df['longitude'] * df['latitude'])

# Features for prediction
features = ['total_rooms', 'area']
X = df[features]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Feature coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: ${coef:,.2f}")
print(f"Intercept: ${model.intercept_:,.2f}")

plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted Values
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Prices ($)')
plt.ylabel('Predicted House Prices ($)')
plt.title('Actual vs Predicted House Prices')

# Plot 2: Total Rooms vs House Price
plt.subplot(1, 3, 2)
plt.scatter(X_test['total_rooms'], y_test, alpha=0.5, label='Actual')
plt.scatter(X_test['total_rooms'], y_pred, alpha=0.5, label='Predicted')
plt.xlabel('Total Rooms')
plt.ylabel('House Price ($)')
plt.title('Total Rooms vs House Price')
plt.legend()

# Plot 3: Geographic Area vs House Price
plt.subplot(1, 3, 3)
plt.scatter(X_test['area'], y_test, alpha=0.5, label='Actual')
plt.scatter(X_test['area'], y_pred, alpha=0.5, label='Predicted')
plt.xlabel('Area (sq degrees)')
plt.ylabel('House Price ($)')
plt.title('Area vs House Price')
plt.legend()

plt.tight_layout()
plt.show()