#!/usr/bin/env python
# coding: utf-8

# In[749]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc


# In[750]:


df=pd.read_csv('train_class.csv')
test=pd.read_csv('test_class.csv')


# In[751]:


df.head()


# In[752]:


df.info()


# In[753]:


df.isnull().sum()


# In[754]:


df['Ever_Married'].fillna('No',inplace=True)
df['Graduated'].fillna('No',inplace=True)


# In[755]:


df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)
df['Var_1'].fillna(df['Var_1'].mode()[0], inplace=True)


# In[756]:


df['Work_Experience'].fillna(df['Work_Experience'].mean(), inplace=True)
df['Family_Size'].fillna(df['Family_Size'].mean(), inplace=True)


# In[757]:


df.isnull().sum()


# In[758]:


pie=df.groupby('Segmentation')[[ 'Age','Work_Experience','Family_Size']].agg('mean')


# In[759]:


pie.plot(kind='bar')


# In[760]:


df.groupby(['Profession','Gender'])[['Gender']].count().plot(kind='bar')


# In[761]:


sns.heatmap(df.corr(),annot = True)


# In[762]:


categorical_col = df.select_dtypes('object')

for i in categorical_col:
    print(df[i].value_counts(), end="\n\n")


# In[763]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
df = pd.get_dummies(df,columns=cat)
df


# In[764]:


X = df.drop(['Segmentation','ID'],axis = 1)
Y = df['Segmentation']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=33)


# In[765]:


model =KNeighborsClassifier()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print(f"The accuracy score for KNeighborsClassifier   is {(accuracy_score(Y_test,y_pred)*100).round(2)}")


# In[766]:


from sklearn import metrics
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))


# In[767]:


leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
knn_2 = KNeighborsClassifier()
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
best_model = clf.fit(X_train,Y_train)
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[768]:


model_grid =KNeighborsClassifier(n_neighbors=7,leaf_size=5,p=1)
model_grid.fit(X_train,Y_train)
y_pred_grid = model_grid.predict(X_test)
print(f"The accuracy score for KNeighborsClassifier   is {(accuracy_score(Y_test,y_pred)*100).round(2)}")


# In[769]:


from sklearn import metrics
print(metrics.classification_report(Y_test, y_pred_grid))
print(metrics.confusion_matrix(Y_test, y_pred_grid))


# In[770]:


test


# In[771]:


test['Ever_Married'].fillna('No',inplace=True)
test['Graduated'].fillna('No',inplace=True)


# In[772]:


test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
test['Var_1'].fillna(test['Var_1'].mode()[0], inplace=True)


# In[773]:



test['Work_Experience'].fillna(test['Work_Experience'].mean(), inplace=True)
test['Family_Size'].fillna(test['Family_Size'].mean(), inplace=True)


# In[774]:


test.isnull().sum()


# In[775]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
test = pd.get_dummies(test,columns=cat)
test


# In[776]:


test_new=test.drop(['ID'],axis=1)
test_new


# In[777]:


prediction=model_grid.predict(test_new)
res=prediction


# In[778]:


results_df= pd.DataFrame({'ID':test.ID ,'Segmentation': res})
results_df


# In[779]:


results_df.to_csv('KNN.csv',index=False)


# In[ ]:




