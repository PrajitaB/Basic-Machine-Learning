# Implement a random forest classifier to predict customer churn in a telecom dataset.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = r'C:\Users\KIIT\Desktop\Lab Experiment\AD Lab\TelcoCustomerChurn (week 6).csv'
data = pd.read_csv(data_path)

# Data preprocessing
data.drop('customerID', axis=1, inplace=True) # Avoid Overfitting

# Missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Categorical variables to numeric
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split dataset into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Results
print("\n")
print(f"Overall Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)