#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import streamlit as str
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_curve, roc_auc_score


# In[2]:


df_train = pd.read_csv('Titanic_train.csv')
df_test = pd.read_csv('Titanic_test.csv')
df_train.head()


# In[3]:


df_test.head()


# In[4]:


# Examining the features.
df_train = df_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_train.head()


# In[5]:


df_test.head()


# ## Types:
"Sex" and "Embarked" are categorical columns and the rest are numerical columns.
# ## Visualisations :

# In[8]:


sns.pairplot(df_train)
plt.show()


# ## 2. Data Preprocessing:

# In[10]:


## Handling missing values:
df_train.isnull().sum()


# In[11]:


df_test.isnull().sum()


# In[12]:


# fill null values with Mode:
df_train['Age'].fillna(st.mode(df_train['Age']), inplace=True)


# In[13]:


df_test['Age'].fillna(st.mode(df_test['Age']), inplace=True)


# In[14]:


df_train['Embarked'].fillna(st.mode(df_train['Embarked']), inplace=True)
df_test['Fare'].fillna(st.mode(df_test['Fare']), inplace=True)


# In[15]:


df_test.isnull().sum()


# In[16]:


df_test.isnull().sum()


# ## Encoding categorical variables.

# In[18]:


lb = LabelEncoder()


# In[19]:


np.unique(df_test['Sex'])


# In[20]:


arr = lb.fit_transform(df_train[['Sex']])
# converted value of female to 0 and male to 1.


# In[21]:


df_train['Sex'] = arr


# In[22]:


arr1 = arr = lb.fit_transform(df_test[['Sex']])
df_test['Sex'] = arr1


# In[23]:


np.unique(df_test['Embarked'])


# In[24]:


arr2 = lb.fit_transform(df_train[['Embarked']]) # ['C', 'Q', 'S'] converted to [0, 1, 2]
df_train['Embarked'] = arr2


# In[25]:


arr3 = lb.fit_transform(df_test[['Embarked']])
df_test['Embarked'] = arr3


# In[26]:


## spliting data:
df_train.head()


# In[27]:


features = df_train.drop('Survived', axis=1)
target = df_train['Survived']


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(features, target)


# ## 3. Model building:

# In[30]:


lr = LogisticRegression()
lr.fit(x_train, y_train)


# In[31]:


# predicting the test data
lr.predict(df_test)


# ## 4. Model Evaluation:

# In[33]:


y_pred = lr.predict(x_test)


# In[34]:


# Evaluating the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score
accuracy_score(y_test,y_pred)*100


# In[35]:


precision_score(y_test,y_pred)*100


# In[36]:


recall_score(y_test,y_pred)*100


# In[37]:


f1_score(y_test,y_pred)*100


# In[38]:


auc_score = roc_auc_score(y_test,y_pred)*100
auc_score


# In[39]:


sigma = lr.predict_proba(x_test)[:,1]
sigma


# In[40]:


# fpr and tpr values with respect to threshold values for visualising roc curves'
# fpr = false positive rate and tpr = true positive rate.
fpr, tpr, threshold = roc_curve(y_test,sigma)
print(fpr,tpr,threshold)


# In[41]:


# visualisation for roc_auc curves:
sns.lineplot(x=fpr, y=tpr, color='r', label=f'AUC_score={auc_score:.2f}%')
plt.plot([0, 1], [0, 1], linestyle='dashed', color='grey')
plt.xlabel('Fpr')
plt.ylabel('Tpr')
plt.title('ROC_curve')
plt.grid()
plt.legend()
plt.show()


# ## 5. Interpretation:
# co-efficient of logistic regression model.

# In[43]:


lr.coef_


# ## 6. Deployment with Streamlit:

# In[45]:


x_train.columns


# In[46]:


## Input from user:
def user_inp():
    pclass = st.sidebar.selectbox('Passenger class : 1st=1, 2nd=2, 3rd=3', [1, 2, 3])
    sex = st.sidebar.selectbox('Gender : male=1, female=0', [0, 1])
    age = st.sidebar.number_input('Enter age :')
    sibsp = st.sidebar.selectbox('How many siblings & spouses of the passenger aboard:', [0, 1, 2, 3, 4, 5, 6, 7, 8])
    parch = st.sidebar.selectbox('Number of parents or children a passenger was traveling with:', [0, 1, 2, 3, 4, 5, 6])
    fare = st.sidebar.number_input('Enter fare :')
    embarked = st.sidebar.selectbox('Port of embarkation : Cherbourg=0, Queenstown=1, Southampton=2', [0, 1, 2])
    data = {'Pclass':pclass, 'Sex':sex, 'Age':age, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare, 'Embarked':embarked}
    features = pd.DataFrame(data, index=[0])
    return features


# In[ ]:


df = user_inp()


# In[ ]:


st.subheader('You entered :')
st.write(df)
st.subheader('Prediction')
y_pred = lr.predict(df)
st.write('Survived' if y_pred==1 else 'Not Survived')
st.subheader('Predict_prob')
pred_prob = lr.predict_proba(df)
st.write(pred_prob)


# ## Interview questions:
Precision :
 out of all instances the model predicted as positive, how many are actually positive.
Recall :
 out of all the actual positive instances, how many were correctly identified by the model.

Cross validation :
 splitting the dataset into multiple subsets and then training and validating the model across these different subsets.
importance:
1.Balances Class Representation
2.Reduces Overfitting
3.Reliable Performance Metrics.
# In[ ]:




