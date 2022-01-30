# regression analysis is a type of predictive modelling , that analyzes relationship between the independant and dependant variable

# linear regression it is to obtain a best fit line through the points,
    # with the least amount of error distance ( distance between *points* to the *line* )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split

ap = pd.read_csv('area-price.csv')
print(ap)

plt.scatter(ap['area'],ap['price'])

model = linear_model.LinearRegression()
x = (ap['area'].values).reshape(-1,1)
print(type(x))
#x.reshape(-1,-1)
y = (ap['price'].values).reshape(-1,1)
print(y)
#.reshape(-1,-1)
# here you are initializing values for train (x,y) values && test (x,y) values
xtrain,xtest,ytrain,ytest = train_test_split(x,y)

#fit the train values to then later find the coefficients
model.fit(xtrain,ytrain)

print(model.coef_)

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

#takes 2d array values
pred = model.predict([[3600]])
scr =model.score(xtest,ytest)
print(scr)