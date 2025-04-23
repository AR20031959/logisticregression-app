
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statistics as st
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load datasets
df_train = pd.read_csv('Titanic_train.csv')
df_test = pd.read_csv('Titanic_test.csv')

# Drop unnecessary columns
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values
df_train['Age'].fillna(st.mode(df_train['Age']), inplace=True)
df_test['Age'].fillna(st.mode(df_test['Age']), inplace=True)
df_train['Embarked'].fillna(st.mode(df_train['Embarked']), inplace=True)
df_test['Fare'].fillna(st.mode(df_test['Fare']), inplace=True)

# Encode categorical variables
lb = LabelEncoder()
df_train['Sex'] = lb.fit_transform(df_train['Sex'])
df_test['Sex'] = lb.fit_transform(df_test['Sex'])
df_train['Embarked'] = lb.fit_transform(df_train['Embarked'])
df_test['Embarked'] = lb.fit_transform(df_test['Embarked'])

# Split data into features and target
features = df_train.drop('Survived', axis=1)
target = df_train['Survived']
x_train, x_test, y_train, y_test = train_test_split(features, target)

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Streamlit user input
def user_input():
    pclass = st.sidebar.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
    sex = st.sidebar.selectbox('Gender (0 = Female, 1 = Male)', [0, 1])
    age = st.sidebar.number_input('Age', min_value=0.0)
    sibsp = st.sidebar.selectbox('Siblings/Spouses Aboard', list(range(9)))
    parch = st.sidebar.selectbox('Parents/Children Aboard', list(range(7)))
    fare = st.sidebar.number_input('Fare', min_value=0.0)
    embarked = st.sidebar.selectbox('Port of Embarkation (0 = C, 1 = Q, 2 = S)', [0, 1, 2])
    
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    return pd.DataFrame(data, index=[0])

# Streamlit app layout
st.title("Titanic Survival Prediction")
df = user_input()

st.subheader("Your Input:")
st.write(df)

st.subheader("Prediction")
prediction = lr.predict(df)
st.write("Prediction:", "Survived" if prediction[0] == 1 else "Not Survived")

st.subheader("Prediction Probability")
prob = lr.predict_proba(df)
st.write(prob)

# Optional: Model performance metrics (for developer/debugging purposes)
y_pred = lr.predict(x_test)
st.sidebar.subheader("Model Performance (on train/test split)")
st.sidebar.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
st.sidebar.write(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
st.sidebar.write(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
st.sidebar.write(f"F1 Score: {f1_score(y_test, y_pred) * 100:.2f}%")
st.sidebar.write(f"AUC Score: {roc_auc_score(y_test, y_pred) * 100:.2f}%")
