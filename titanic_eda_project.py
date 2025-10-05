
# Exploratory Data Analysis (EDA) – Beginner Project
# Dataset: Titanic (sample data)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic_sample_dataset.csv")

# First look at dataset
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Drop rows with missing age values
df_clean = df.dropna(subset=["age"])

# Univariate Analysis - Age Distribution
sns.histplot(df_clean['age'], kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

# Bivariate Analysis - Age vs Survival
sns.boxplot(x="survived", y="age", data=df_clean)
plt.title("Age vs Survival")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Sample Titanic Dataset)")
plt.show()
