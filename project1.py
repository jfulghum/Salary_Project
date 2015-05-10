# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:22:11 2015

@author: johanna
"""

#import pandas as pd
#df = pd.read_csv('C:\Users\johanna\Documents\Kaggle\weather.xlsx',
                # names=['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type'],
                 #index_col='id')
import pandas as pd               
train = pd.read_csv('https://raw.githubusercontent.com/ajschumacher/gadsdata/master/salary/train.csv')

train.shape
train.head()
train.tail()
train.columns
train.describe()

#how many unique values are there for each category? eg sourcename, etc. 
#do for all categorical variables. see about patterns. 
#dummy variable. model must have #s. 
# pick a variable with a high correlation with resposnse (salary) 

train.Category.describe()
train.groupby('SourceName').SalaryNormalized.mean().order()

train[train.SourceName=='OilCareers.com'] #class 3 has good code 


train.Company.value_counts()
train.groupby('Company').SalaryNormalized.mean()

train.groupby('ContractType').SalaryNormalized.mean() 
train.groupby('ContractTime').SalaryNormalized.mean() 

train.ContractType.describe() #count is # of non-null , top = top value, freq is value of that top value
train.ContractTime.describe()

train.isnull().sum()
train.ContractTime.fillna('was_na', inplace=True) # we are replacing nulls with a new value
train.ContractTime.value_counts() #now we have 3 real values. what is the salary by contract type?

train.ContractType.fillna('was_na', inplace=True)
train.ContractType.value_counts()

pd.get_dummies(train.ContractType, prefix='Type').head()

train_dummies = pd.get_dummies(train.ContractType, prefix='Type').iloc[:, 1:]
train_dummies.head()

train = pd.concat([train, train_dummies], axis=1)
train.head()

train_dummies2 = pd.get_dummies(train.ContractTime, prefix='Time').iloc[:, 1:]
train_dummies2.head()

train = pd.concat([train, train_dummies2], axis=1)
train.head()

#next step. split data into train and test, split and run linear model and see what happens.
from sklearn.datasets import load_iris
iris = load_iris()

# create X (features) and y (response)

columns = ['Type_part_time','Type_was_na','Time_permanent','Time_was_na']
X = train[columns]
y = train.SalaryNormalized

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

#import class, instatiaite, fit, predict. thats that same pattern for every model. 
from sklearn.linear_model import LinearRegression #import class
ln = LinearRegression() #create instance of linear regression
ln.fit(X_train, y_train) # fit. its learned the co-efficents. 
ln.coef_  #tells us that compared to baseline(which was fulltime), -6996 and -1221 decreases in salary. 
y_pred = ln.predict(X_test)
print y_test

from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred) #root error is more inrepratable. take sq root. use numpy. np.sqrt
import numpy as np

np.sqrt(metrics.mean_squared_error(y_test,y_pred)) # this is our root mean sqed error. THE goal is to drive that number as low possile. 
#main task = create more features.
#take category and trun that into dummy variables. add those as features. 
#use company? see if that makes sense.
#we know that location is a useful predictor. how to? this will involve regular expressions. 
#decripstion has huge impact on salary. how to take that and make a prediction. 
#conceptually, how to take dsecribtion. 

 
#.com versus .uk? write a list of things to look into if i have time. 
#what are my theories? is saying .com vs .uk a good predictor of salary?
#you need to build your own features and create dummy variables. 
