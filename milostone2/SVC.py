#!/usr/bin/env python
# coding: utf-8

# In[206]:


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


# In[207]:


df=pd.read_csv('train_class.csv')
test=pd.read_csv('test_class.csv')


# In[208]:


df.head()


# In[209]:


df.info()


# In[210]:


df.isnull().sum()


# In[211]:


df['Ever_Married'].fillna('No',inplace=True)
df['Graduated'].fillna('No',inplace=True)


# In[212]:


df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)
df['Var_1'].fillna(df['Var_1'].mode()[0], inplace=True)


# In[213]:


df['Work_Experience'].fillna(df['Work_Experience'].mean(), inplace=True)
df['Family_Size'].fillna(df['Family_Size'].mean(), inplace=True)


# In[214]:


df.isnull().sum()


# In[215]:


pie=df.groupby('Segmentation')[[ 'Age','Work_Experience','Family_Size']].agg('mean')


# In[216]:


pie.plot(kind='bar')


# In[217]:


df.groupby(['Profession','Gender'])[['Gender']].count().plot(kind='bar')


# In[218]:


sns.heatmap(df.corr(),annot = True)
df


# In[219]:


categorical_col = df.select_dtypes('object')

for i in categorical_col:
    print(df[i].value_counts(), end="\n\n")


# In[220]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
df = pd.get_dummies(df,columns=cat)
df


# In[221]:


X = df.drop(['Segmentation','ID'],axis = 1)
Y = df['Segmentation']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=33)


# In[222]:


model =SVC()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print(f"The accuracy score for SVC  is {(accuracy_score(Y_test,y_pred)*100).round(2)}")


# In[223]:


from sklearn import metrics
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))


# In[224]:


#Evaluation 


# In[225]:


from sklearn.model_selection import GridSearchCV
 
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
#grid.fit(X_train, Y_train)


# In[226]:


#print(grid.best_params_)


# In[239]:


SVC_grid =SVC(C=100,gamma=0.001,kernel='rbf')
SVC_grid.fit(X_train,Y_train)
y_pred_grid = SVC_grid.predict(X_test)
print(f"The accuracy score for SVC  is {(accuracy_score(Y_test,y_pred_grid)*100).round(2)}")


# In[228]:


from sklearn import metrics
print(metrics.classification_report(Y_test, y_pred_grid))
print(metrics.confusion_matrix(Y_test, y_pred_grid))


# In[229]:


test


# In[230]:


test['Ever_Married'].fillna('No',inplace=True)
test['Graduated'].fillna('No',inplace=True)


# In[231]:


test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
test['Var_1'].fillna(test['Var_1'].mode()[0], inplace=True)


# In[232]:



test['Work_Experience'].fillna(test['Work_Experience'].mean(), inplace=True)
test['Family_Size'].fillna(test['Family_Size'].mean(), inplace=True)


# In[233]:


test.isnull().sum()


# In[234]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
test = pd.get_dummies(test,columns=cat)
test


# In[235]:


test_new=test.drop(['ID'],axis=1)
test_new


# In[236]:


prediction=SVC_grid.predict(test_new)
res=prediction


# In[237]:


results_df= pd.DataFrame({'ID':test.ID ,'Segmentation': res})
results_df


# In[238]:


results_df.to_csv('SVC.csv',index=False)


# In[ ]:




