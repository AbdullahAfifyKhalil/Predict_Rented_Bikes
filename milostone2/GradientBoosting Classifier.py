#!/usr/bin/env python
# coding: utf-8

# In[895]:


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


# In[896]:


df=pd.read_csv('train_class.csv')
test=pd.read_csv('test_class.csv')


# In[897]:


df.head()


# In[898]:


df.info()


# In[899]:


df.isnull().sum()


# In[900]:


df['Ever_Married'].fillna('No',inplace=True)
df['Graduated'].fillna('No',inplace=True)


# In[901]:


df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)
df['Var_1'].fillna(df['Var_1'].mode()[0], inplace=True)


# In[902]:


df['Work_Experience'].fillna(df['Work_Experience'].mean(), inplace=True)
df['Family_Size'].fillna(df['Family_Size'].mean(), inplace=True)


# In[903]:


df.isnull().sum()


# In[904]:


pie=df.groupby('Segmentation')[[ 'Age','Work_Experience','Family_Size']].agg('mean')


# In[905]:


pie.plot(kind='bar')


# In[906]:


df.groupby(['Profession','Gender'])[['Gender']].count().plot(kind='bar')


# In[907]:


sns.heatmap(df.corr(),annot = True)


# In[908]:


categorical_col = df.select_dtypes('object')

for i in categorical_col:
    print(df[i].value_counts(), end="\n\n")


# In[909]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
df = pd.get_dummies(df,columns=cat)
df


# In[910]:


X = df.drop(['Segmentation','ID'],axis = 1)
Y = df['Segmentation']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=33)


# In[911]:


model=GradientBoostingClassifier()
#model =GradientBoostingClassifier(n_estimators=1000,max_depth=10,min_samples_leaf=4,min_samples_split=10)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print(f"The accuracy score for GradientBoosting Classifier   is {(accuracy_score(Y_test,y_pred)*100).round(2)}")


# In[912]:


from sklearn import metrics
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))


# In[ ]:





# In[913]:


test


# In[914]:


test['Ever_Married'].fillna('No',inplace=True)
test['Graduated'].fillna('No',inplace=True)


# In[915]:


test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
test['Var_1'].fillna(test['Var_1'].mode()[0], inplace=True)


# In[916]:



test['Work_Experience'].fillna(test['Work_Experience'].mean(), inplace=True)
test['Family_Size'].fillna(test['Family_Size'].mean(), inplace=True)


# In[917]:


test.isnull().sum()


# In[918]:


cat = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
test = pd.get_dummies(test,columns=cat)
test


# In[919]:


test_new=test.drop(['ID'],axis=1)
test_new


# In[920]:


prediction=model_grid.predict(test_new)
res=prediction


# In[921]:


results_df= pd.DataFrame({'ID':test.ID ,'Segmentation': res})
results_df


# In[922]:


results_df.to_csv('GradBoost.csv',index=False)


# In[ ]:




