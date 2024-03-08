import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Change the path to your dataset
data = pd.read_csv('C:\\Users\\90541094959\\Downloads\\Data.csv')

# Basic analysis (head, tail, describe)
print("First 5 rows:\n", data.head())
print("\nLast 5 rows:\n", data.tail())
print("\nDescriptive statistics:\n", data.describe())

# Visualization
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Distribution of Customer Tenure
sns.histplot(data['tenure'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Customer Tenure')

# Age Distribution
sns.histplot(data['age'], bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution')

# Income Distribution
sns.histplot(data['income'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Income Distribution')

# Education Level Distribution
sns.countplot(x='ed', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Education Level Distribution')

plt.tight_layout()
plt.show()
