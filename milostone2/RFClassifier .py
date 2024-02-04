#!/usr/bin/env python
# coding: utf-8

# In[4157]:


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


# In[4158]:


df=pd.read_csv('train_class.csv')
test=pd.read_csv('test_class.csv')


# In[4159]:


df.head()


# In[4160]:


df.info()


# In[4161]:


df.isnull().sum()


# In[4162]:


df['Ever_Married'].fillna('No',inplace=True)
df['Graduated'].fillna('No',inplace=True)


# In[4163]:


df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)
df['Var_1'].fillna(df['Var_1'].mode()[0], inplace=True)


# In[4164]:


df['Work_Experience'].fillna(df['Work_Experience'].mean(), inplace=True)
df['Family_Size'].fillna(df['Family_Size'].mean(), inplace=True)


# In[4165]:


df.isnull().sum()


# In[4166]:


pie=df.groupby('Segmentation')[[ 'Age','Work_Experience','Family_Size']].agg('mean')


# In[4167]:


pie.plot(kind='bar')


# In[4168]:


df.groupby(['Profession','Gender'])[['Gender']].count().plot(kind='bar')


# In[4169]:


sns.heatmap(df.corr(),annot = True)


# In[4170]:


df


# In[4171]:


categorical_col = df.select_dtypes('object')

for i in categorical_col:
    print(df[i].value_counts(), end="\n\n")


# In[4172]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
df = pd.get_dummies(df,columns=cat)
df


# In[4173]:


X = df.drop(['Segmentation','ID'],axis = 1)
Y = df['Segmentation']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=33)


# In[4174]:


model =RandomForestClassifier()
model.fit(X_train,Y_train)
y_pred_rf = model.predict(X_test)
print(f"The accuracy score for RFClassifier  is {(accuracy_score(Y_test,y_pred_rf)*100).round(2)}")


# In[4176]:


''''param_grid = { 
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
   ' min_samples_split' :[2, 5, 10, 15, 100],
    'min_samples_leaf':[1, 2, 5, 10],
    'bootstrap':[True, False],
    'max_features':['auto', 'sqrt'],
}'''
param_grid = { 
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split' :[2, 5, 10, 15, 100],
    'min_samples_leaf':[1, 2, 5, 10],
    'max_depth' :[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'bootstrap':[True, False],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y_train)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


#model =RandomForestClassifier(n_estimators=400,max_depth=70,min_samples_leaf=4,min_samples_split=10,bootstrap=True,max_features='auto')

rf_grid = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=5,criterion='',
                           bootstrap=   , max_features=' ' )
rf_grid.fit(X_train,Y_train)
y_pred_rf = rf_grid.predict(X_test)
print(f"The accuracy score for Random Forest is {(accuracy_score(Y_test,y_pred_rf)*100).round(2)}")


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(Y_test, y_pred_rf))
print(metrics.confusion_matrix(Y_test, y_pred_rf))


# In[ ]:


test


# In[ ]:


test['Ever_Married'].fillna('No',inplace=True)
test['Graduated'].fillna('No',inplace=True)


# In[ ]:


test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
test['Var_1'].fillna(test['Var_1'].mode()[0], inplace=True)


# In[ ]:


test['Work_Experience'].fillna(test['Work_Experience'].mean(), inplace=True)
test['Family_Size'].fillna(test['Family_Size'].mean(), inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
test = pd.get_dummies(test,columns=cat)
test


# In[ ]:


test_new=test.drop(['ID'],axis=1)
test_new


# In[ ]:


prediction=rf_grid.predict(test_new)
res=prediction


# In[ ]:


results_df= pd.DataFrame({'ID':test.ID ,'Segmentation': res})
results_df


# In[ ]:


results_df.to_csv('RFClassifier .csv',index=False)

