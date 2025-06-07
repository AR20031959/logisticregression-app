#!/usr/bin/env python
# coding: utf-8

# In[158]:


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


# In[159]:


import pandas as pd


df_train = pd.read_csv(r'D:\EXCEL R\Data Science\Assignments\6. Logistic Regression\Logistic Regression\Titanic_train.csv')
df_test = pd.read_csv(r'D:\EXCEL R\Data Science\Assignments\6. Logistic Regression\Logistic Regression\Titanic_test.csv')

df_train.head()


# In[160]:


df_test.head()


# In[161]:


# Examining the features.
df_train = df_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_train.head()


# In[162]:


df_test.head()


# ## Types:
"Sex" and "Embarked" are categorical columns and the rest are numerical columns.
# ## Visualisations :

# In[163]:


sns.pairplot(df_train)
plt.show()


# ## 2. Data Preprocessing:

# In[164]:


## Handling missing values:
df_train.isnull().sum()


# In[165]:


df_test.isnull().sum()


# In[166]:


# fill null values with Mode:
df_train['Age'].fillna(st.mode(df_train['Age']), inplace=True)


# In[167]:


df_test['Age'].fillna(st.mode(df_test['Age']), inplace=True)


# In[168]:


df_train['Embarked'].fillna(st.mode(df_train['Embarked']), inplace=True)
df_test['Fare'].fillna(st.mode(df_test['Fare']), inplace=True)


# In[169]:


df_test.isnull().sum()


# In[170]:


df_test.isnull().sum()


# ## Encoding categorical variables.

# In[171]:


lb = LabelEncoder()


# In[172]:


np.unique(df_test['Sex'])


# In[173]:


arr = lb.fit_transform(df_train[['Sex']])
# converted value of female to 0 and male to 1.


# In[174]:


df_train['Sex'] = arr


# In[175]:


arr1 = arr = lb.fit_transform(df_test[['Sex']])
df_test['Sex'] = arr1


# In[176]:


np.unique(df_test['Embarked'])


# In[177]:


arr2 = lb.fit_transform(df_train[['Embarked']]) # ['C', 'Q', 'S'] converted to [0, 1, 2]
df_train['Embarked'] = arr2


# In[178]:


arr3 = lb.fit_transform(df_test[['Embarked']])
df_test['Embarked'] = arr3


# In[179]:


## spliting data:
df_train.head()


# In[180]:


features = df_train.drop('Survived', axis=1)
target = df_train['Survived']


# In[181]:


x_train, x_test, y_train, y_test = train_test_split(features, target)


# ## 3. Model building:

# In[182]:


lr = LogisticRegression()
lr.fit(x_train, y_train)


# In[183]:


# predicting the test data
lr.predict(df_test)


# ## 4. Model Evaluation:

# In[184]:


y_pred = lr.predict(x_test)


# In[185]:


# Evaluating the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score
accuracy_score(y_test,y_pred)*100


# In[186]:


precision_score(y_test,y_pred)*100


# In[187]:


recall_score(y_test,y_pred)*100


# In[188]:


f1_score(y_test,y_pred)*100


# In[189]:


auc_score = roc_auc_score(y_test,y_pred)*100
auc_score


# In[190]:


sigma = lr.predict_proba(x_test)[:,1]
sigma


# In[191]:


# fpr and tpr values with respect to threshold values for visualising roc curves'
# fpr = false positive rate and tpr = true positive rate.
fpr, tpr, threshold = roc_curve(y_test,sigma)
print(fpr,tpr,threshold)


# In[192]:


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

# In[193]:


lr.coef_


# ## 6. Deployment with Streamlit:

# In[194]:


x_train.columns


# In[195]:


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


# In[198]:


arr = lb.fit_transform(df_train['Sex'])
# converted value of female to 0 and male to 1.

arr1 = lb.fit_transform(df_test['Sex'])
df_test['Sex'] = arr1

arr2 = lb.fit_transform(df_train['Embarked']) # ['C', 'Q', 'S'] converted to [0, 1, 2]
df_train['Embarked'] = arr2

arr3 = lb.fit_transform(df_test['Embarked'])
df_test['Embarked'] = arr3


# In[203]:


import streamlit as st

input_df = user_inp()
st.subheader('You entered :')
st.write(input_df)
st.subheader('Prediction')
y_pred = lr.predict(input_df)
st.write('Survived' if y_pred[0]==1 else 'Not Survived')
st.subheader('Predict_prob')
pred_prob = lr.predict_proba(input_df)
st.write(pred_prob)


# In[ ]:





# In[ ]:





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




