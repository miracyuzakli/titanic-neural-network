# Titanic Data Exploration and Analysis

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
data = pd.read_csv("data/titanic.csv")

# Display basic statistics of the dataset
print(data.describe())

# Visualize Survival Counts
sns.set(style="whitegrid")
sns.countplot(x='Survived', data=data, palette='Set1')
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()

# Pairplot for numerical features with hue based on Survival
sns.pairplot(data, hue='Survived', diag_kind='kde')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
