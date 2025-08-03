# Student Score Analysis: Multiple linear regression analysis to handle multiple predictors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df):
    df['attendance'] = 10 - df['absence_days']
    df['study_hours'] = df['weekly_self_study_hours']
    subjects = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
    df['assignment_scores'] = df[subjects].mean(axis=1)
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df

def train_model(df):
    features = ['attendance', 'study_hours']
    X = df[features]
    y = df['assignment_scores'] # Target : deoendent variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred, features

def evaluate_model(model, features, X_test, y_test, y_pred):

    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Summary:")
    print("Features used for prediction:")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: coefficient = {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, rmse, r2

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual vs predicted values
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Actual Data')
    
    line_min = min(min(y_test), min(y_pred))
    line_max = max(max(y_test), max(y_pred))
    plt.plot([line_min, line_max], [line_min, line_max], 
             'r--', lw=2, label='Regression Line')
    
    plt.xlabel('Actual Assignment Scores')
    plt.ylabel('Predicted Assignment Scores')
    plt.title('Student Performance: Actual vs Predicted Scores')
    plt.legend()
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
    plt.text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv('student-scores (week 3).csv')
        
    print("Dataset Summary:")
    print(df.describe())
        
        # Prepare
    df = prepare_data(df)
        
        # Training
    model, X_train, X_test, y_train, y_test, y_pred, features = train_model(df)
        
        # Evaluation
    mse, rmse, r2 = evaluate_model(model, features, X_test, y_test, y_pred)
        
        # Scatter Plot
    plot_results(y_test, y_pred)
        

if __name__ == "__main__":
    main()