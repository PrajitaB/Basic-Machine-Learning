# Basic Data Preprocessing and Visualization using Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 2. Load a sample dataset using pandas (e.g., Iris or a custom dataset).
file_path = r"C:\Users\KIIT\Desktop\Lab Experiment\AD Lab\IRIS (week 1,5).csv"
df = pd.read_csv(file_path)

# Display
print("Original Data:")
print(df.head())

# Missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 1.1 Demonstrate data preprocessing steps: Handling missing values (mean)
imputer = SimpleImputer(strategy='mean')
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = imputer.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

# 1.1 Demonstrate data preprocessing steps: Handling missing values (Most Frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['species']] = cat_imputer.fit_transform(df[['species']])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# 1.2 Demonstrate data preprocessing steps: Encoding categorical data
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

print("\nData After Encoding Categorical Data:")
print(df.head())

# 1.3 Demonstrate data preprocessing steps: Feature scaling using StandardScaler(Z)
scaler = StandardScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

print("\nData After Feature Scaling:")
print(df.head())

# 3. Plot the distribution of a feature (sepal_length) using matplotlib.pyplot.hist().
plt.figure(figsize=(8, 5))
plt.hist(df['sepal_length'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Scaled Sepal Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. Create scatter plots to understand relationships between features using seaborn.scatterplot().
# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, palette='deep')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.legend(title='Species')
plt.show()

# 5. Use a correlation heatmap to find the relationship between multiple features with seaborn.heatmap().
# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()