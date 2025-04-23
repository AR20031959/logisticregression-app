
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Load datasets
df_train = pd.read_csv('Titanic_train.csv')
df_test = pd.read_csv('Titanic_test.csv')

# Drop unnecessary columns
df_train = df_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

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

# Feature-target split
features = df_train.drop('Survived', axis=1)
target = df_train['Survived']
x_train, x_test, y_train, y_test = train_test_split(features, target)

# Model training
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Streamlit interface
def user_inp():
    pclass = st.sidebar.selectbox('Passenger class : 1st=1, 2nd=2, 3rd=3', [1, 2, 3])
    sex = st.sidebar.selectbox('Gender : male=1, female=0', [0, 1])
    age = st.sidebar.number_input('Enter age :')
    sibsp = st.sidebar.selectbox('Siblings/Spouses aboard:', list(range(9)))
    parch = st.sidebar.selectbox('Parents/Children aboard:', list(range(7)))
    fare = st.sidebar.number_input('Enter fare :')
    embarked = st.sidebar.selectbox('Port of embarkation : Cherbourg=0, Queenstown=1, Southampton=2', [0, 1, 2])
    data = {'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_inp()

st.subheader('You entered :')
st.write(df)

st.subheader('Prediction')
y_pred = lr.predict(df)
st.write('Survived' if y_pred[0] == 1 else 'Not Survived')

st.subheader('Prediction Probability')
pred_prob = lr.predict_proba(df)
st.write(pred_prob)

# Interview notes (as comments for internal learning)
# Precision: Out of all instances the model predicted as positive, how many are actually positive.
# Recall: Out of all the actual positive instances, how many were correctly identified by the model.
# Cross-validation: Splitting the dataset into multiple subsets and then training/validating across these subsets.
# Importance:
# 1. Balances Class Representation
# 2. Reduces Overfitting
# 3. Reliable Performance Metrics
