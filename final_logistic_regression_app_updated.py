
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import joblib

# Load datasets
df_train = pd.read_csv(r"C:\\Users\\HP\\New_codes_python\\logistic regression\\Titanic_train.csv")
df_test = pd.read_csv(r"C:\\Users\\HP\\New_codes_python\\logistic regression\\Titanic_test.csv")

# Drop irrelevant columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
df_test.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Fill missing values
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

# Encode categorical variables
le = LabelEncoder()
for col in ['Sex', 'Embarked']:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Split train data
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")  # Save the figure
plt.close()

# Interpret coefficients
print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_[0]):
    print(f"{col}: {coef:.4f}")

# Save model
joblib.dump(model, "logistic_model.pkl")

# Streamlit App
def run_streamlit():
    st.title("Titanic Survival Prediction")

    Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    Sex = st.selectbox("Sex", ["female", "male"])
    Age = st.slider("Age", 0, 100, 30)
    SibSp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.slider("Parents/Children Aboard", 0, 6, 0)
    Fare = st.number_input("Fare Paid", min_value=0.0, value=32.2)
    Embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

    input_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [0 if Sex == "female" else 1],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [le.transform([Embarked])[0]]
    })

    model = joblib.load("logistic_model.pkl")
    prediction = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")
    st.write("Survived" if prediction == 1 else "Did Not Survive")
    st.write(f"Survival Probability: {pred_proba:.2f}")

if __name__ == "__main__":
    run_streamlit()
