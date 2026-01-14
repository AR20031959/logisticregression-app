
# ============================================================
# model_trainer.py
# Train Logistic Regression model and save as titanic_model.pkl
# ============================================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load datasets (adjust path if needed)
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# Handle missing values
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)

train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
test_df["Embarked"].fillna(test_df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
train_df = pd.get_dummies(train_df, columns=["Sex", "Embarked"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["Sex", "Embarked"], drop_first=True)

# Features and target
X = train_df[["Pclass", "Age", "SibSp", "Parch", "Fare",
              "Sex_male", "Embarked_Q", "Embarked_S"]]
y = train_df["Survived"]

# Train model
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "titanic_model.pkl")

print("Model trained and saved as titanic_model.pkl")
