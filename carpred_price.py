# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 09:59:29 2022

@author: HP notebook
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
os.chdir('D:\PYTHON FOR DATA SCIENCE -NPTEL\WEEK  5')

car_data=pd.read_csv("quiker_car.csv")
car=car_data.copy()
print(car.info())
print(car.isnull().sum())
## CLEANING DATASET

# modifying car year
car=car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)

# modifying car price
car['Price'].unique()
car=car[car['Price']!="Ask For Price"]
car['Price']=car['Price'].str.replace(',','')
car['Price']=car['Price'].astype(int)
# modifying kms driven
car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0) #split the string between space and get first word
car['kms_driven']=car['kms_driven'].str.replace(',','') #replace , wuth empty string
car=car[car['kms_driven'].str.isnumeric()] # keep only rows with numeric string
car['kms_driven']=car['kms_driven'].astype(int)
# modifying fuel type
car=car[~car['fuel_type'].isna()]
# modifying fuel type
car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')

# RESET the index
car.reset_index(drop=True)
del car['Unnamed: 0']


## plots
sns.lmplot(x='kms_driven',y='Price',data=car,hue='fuel_type',legend=True)
sns.boxplot(x=car['fuel_type'],y=car['Price'])

## removing and detecting outliers
car.describe()
car=car[car['Price']<2e6].reset_index(drop=True)
## exporting cleaned data to csv file
car.to_csv('Cleaned_cardata.csv')


### Converting categorical values to dummy variables
car_dum=pd.get_dummies(car,drop_first=True)

### MODEL BUILDING

# assigning x y
x=car_dum.drop(columns='Price')
y=car_dum['Price']
# importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# splitting into train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

### FITTING LINEAR MODEL AND PREDICTION
lgr=LinearRegression(fit_intercept=True)
model_lin=lgr.fit(x_train,y_train)
car_pred=lgr.predict(x_test).round(2)


# R square,MSE and RSME 
MSE=mean_squared_error(y_test,car_pred)
RSME=np.sqrt(MSE)
print(RSME)
r2_test=model_lin.score(x_test,y_test)
r2_train=model_lin.score(x_train,y_train)
print(r2_test,r2_train)

## Residuals Analysis
residuals=y_test-car_pred
sns.regplot(x=car_pred,y=residuals)
residuals.describe()
sns.displot(residuals)


