# Develop a decision tree model to classify species in the Iris dataset.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = r"C:\Users\KIIT\Desktop\Lab Experiment\AD Lab\IRIS (week 1,5).csv"
df = pd.read_csv(dataset_path)

print("Dataset:")
print(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:, -1] 

# Split features (X) and labels (y)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(15, 10))
plot_tree(
    dt_model,
    feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    class_names=dt_model.classes_,
    filled=True,
    rounded=True,
    fontsize=12,
)
plt.title("Decision Tree:", fontsize=12)
plt.tight_layout()
plt.show()