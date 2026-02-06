# Core
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings("ignore")

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
    f1_score
)


df = pd.read_csv("Sales.csv")
df.head()
df.tail()
df.shape
df.info()
df.describe()

df.isnull().sum()

# Fill with mean (only numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill with mode (for others)
df.fillna(df.mode().iloc[0], inplace=True)

# Drop rows if any still null (unlikely after fill)
df.dropna(inplace=True)

df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Outlier removal on Price
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df["Price"] < (Q1 - 1.5*IQR)) | 
          (df["Price"] > (Q3 + 1.5*IQR)))]

le = LabelEncoder()
# Use Country as the categorical column
df["Country_Encoded"] = le.fit_transform(df["Country"])

# One hot encoding example
# df = pd.get_dummies(df, columns=["Country"], drop_first=True)

# Define X for scaling (using numeric features)
X = df[["Price", "Quantity", "Country_Encoded"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Correlation matrix (only numeric)
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr())

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.savefig("heatmap.png")
plt.show()

# Histogram
numeric_df.hist(figsize=(10,10))
plt.savefig("histograms.png")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x=df["Price"])
plt.savefig("boxplot.png")
plt.show()

# Scatterplot
plt.figure()
sns.scatterplot(x="Price", y="Quantity", data=df)
plt.savefig("scatterplot.png")
plt.show()

print("Script completed successfully. Plots saved.")

