import pandas as pd

df = pd.read_csv("car data.csv")

print(df.columns.tolist())
print(df.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("car data.csv")

# Display first 5 rows
print(df.head())

# Create Car Age
df["Car_Age"] = 2024 - df["Year"]

# Remove unnecessary columns
df.drop(["Car_Name","Year"], axis=1, inplace=True)

# Convert categorical data
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# Compare Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(result.head())

# Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()