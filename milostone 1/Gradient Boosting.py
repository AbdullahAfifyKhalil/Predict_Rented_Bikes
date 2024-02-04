import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime
import statsmodels.formula.api as sm

rent_data=pd.read_csv('train.csv',parse_dates=['Date'])
test=pd.read_csv('test.csv',parse_dates=['Date'])

rent_data.info()

rent_data.replace({'Rainfall(mm)':{0:2}}, inplace=True)
rent_data.replace({'Snowfall (cm)':{0:1}}, inplace=True)

rent_data.describe().round(2)

rent_data['Date']=pd.to_datetime(rent_data['Date'],format="%d/%m/%Y")
rent_data['Month']=rent_data['Date'].dt.month
rent_data['Year']=rent_data['Date'].dt.year
rent_data['WeekDay']=rent_data["Date"].dt.day
rent_data

corr_matrix=rent_data.corr()

fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(corr_matrix, annot=True, linewidths=.6)
plt.show()

df1Corr=pd.DataFrame(rent_data.corr().unstack().sort_values(ascending=False)['Rented Bike Count'],columns=['Correlation to the target'])
df1Corr.style.background_gradient(cmap=sns.light_palette("red", as_cmap=True))

rent_data.skew().sort_values(ascending=True) # Snowfall and Rainfall are highly skewed

rent_data['label_day_night']=rent_data['Hour'].apply(lambda x : 'Night' if (x >20 or x<5) else( 'Day'))
test['label_day_night']=test['Hour'].apply(lambda x : 'Night' if (x >20 or x<5) else( 'Day'))

Seasons_vis=pd.DataFrame(rent_data.groupby('Seasons').sum()['Rented Bike Count'].sort_values(ascending=False))
Seasons_vis

rent_data.groupby('Solar Radiation (MJ/m2)').mean()['Rented Bike Count'].plot()
rent_data.groupby('Hour').sum()['Rented Bike Count'].plot()
rent_data.groupby('label_day_night').sum()['Rented Bike Count'].plot.pie()
rent_data.groupby('Temperature(°C)').mean()['Rented Bike Count'].plot()
rent_data.groupby('Holiday').sum()['Rented Bike Count'].plot.pie()

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
rent_data['Seasons']= label_encoder.fit_transform(rent_data['Seasons'])
rent_data['Functioning Day']= label_encoder.fit_transform(rent_data['Functioning Day'])
rent_data['Holiday']= label_encoder.fit_transform(rent_data['Holiday'])
test['Holiday']= label_encoder.fit_transform(test['Holiday'])
test['Functioning Day']= label_encoder.fit_transform(test['Functioning Day'])
rent_data['label_day_night']= label_encoder.fit_transform(rent_data['label_day_night'])
test['label_day_night']= label_encoder.fit_transform(test['label_day_night'])

rent_data.skew().sort_values(ascending=True) 

plt.figure(figsize=(5, 10))
heatmap = sns.heatmap( rent_data.corr()[['Rented Bike Count']].sort_values(by='Rented Bike Count', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Rented Bike Count', fontdict={'fontsize':18}, pad=16);

X=rent_data.drop(['Rented Bike Count','Dew point temperature(°C)','Date','Year'],axis=1)
Y=rent_data['Rented Bike Count']
X

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size = 0.2, random_state = 42)
X_test.shape

from sklearn.metrics import mean_squared_error
from math import sqrt
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size = 0.2,random_state=42)
reg=GradientBoostingRegressor()
reg.fit(X_train, y_train)
y_pred=reg.predict (X_test)
print(sqrt(mean_squared_error(y_test, y_pred)))

from statsmodels.tsa.arima.model import  #time series
model = ARIMA(Y, order=(1,1,0))
model_fit = model.fit()
print(model_fit.summary())

test['Date']=pd.to_datetime(test['Date'],format="%d/%m/%Y")
test['Month']=test['Date'].dt.month
test['Year']=test['Date'].dt.year
test['WeekDay']=test["Date"].dt.day
test

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
test['Seasons']= label_encoder.fit_transform(test['Seasons'])

test_new=test.drop(['ID','Dew point temperature(°C)','Date','Year'],axis=1)

test_new.to_csv('TESTZ.csv',index=False)
test_new

prediction=reg.predict(test_new)
res=prediction

results_df= pd.DataFrame({'ID':test.ID ,'Rented Bike Count': res})
results_df

results_df.to_csv('GradBoo.csv',index=False)