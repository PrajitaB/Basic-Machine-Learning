import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r'C:\Users\KIIT\Desktop\Lab Experiment\AD Lab\IRIS.csv'
df = pd.read_csv(file_path)

# Rename columns
df.rename(columns={
    'sepal_length': 'Area',
    'sepal_width': 'Rooms',
    'petal_length': 'Price'
}, inplace=True)

# 1. Simple linear regression function
def linear_regression(X, y, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Details ({feature_name}):")
    print(f"Coefficient (Slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return X_test, y_test, y_pred

# 2. Scatter plot with regression line
def scatter_plot(X_test_area, y_test_area, y_pred_area, 
                    X_test_rooms, y_test_rooms, y_pred_rooms):
    plt.figure(figsize=(12, 6))

    # For Area
    plt.subplot(1, 2, 1)
    plt.scatter(X_test_area, y_test_area, color='blue', label='Actual Data')
    plt.plot(X_test_area, y_pred_area, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Linear Regression: Area vs Price')
    plt.legend()

    # For Number of rooms
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_rooms, y_test_rooms, color='purple', label='Actual Data')
    plt.plot(X_test_rooms, y_pred_rooms, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Rooms')
    plt.ylabel('Price')
    plt.title('Linear Regression: Rooms vs Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

X_area = df[['Area']]
y_price = df['Price']
X_test_area, y_test_area, y_pred_area = linear_regression(X_area, y_price, 'Area')

X_rooms = df[['Rooms']]
X_test_rooms, y_test_rooms, y_pred_rooms = linear_regression(X_rooms, y_price, 'Number of Rooms')

scatter_plot(X_test_area, y_test_area, y_pred_area, 
                X_test_rooms, y_test_rooms, y_pred_rooms)