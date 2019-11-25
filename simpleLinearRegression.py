# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:20:03 2019

@author: nohaw
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

#pandas read csv
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
#pandas.DataFrame.head->return top n (5 by default) rows of a data frame
df.head()

# summarize the data->is used to view some series of numeric values
df.describe()

#show these columns with 9 rows of data
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# A histogram shows the frequency on the y-axis and range on x-axis
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


#A scatter plot is a type of plot that shows the data as a collection of points.
#The position of a point depends on its two-dimensional value, 
#where each value is a position on either the horizontal or vertical dimension.

#lets plot each of these features vs the Emission, to see how linear is their relation.

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="red")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()


#make Train/Test Split technique to
#provide a more accurate evaluation on out-of-sample accuracy

#Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. 
#We create a mask to select random rows using np.random.rand() function: 

#This function returns Random values in a given shape. It Create an array of the given shape
#and populate it with random samples from a uniform distribution over [0, 1).
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#linear regression model
regr = linear_model.LinearRegression()

# convert input to an array
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

#.fit(), you calculate the optimal values of the weights theta0 and theta1
regr.fit (train_x, train_y)

# The coefficients
#theta1
print ('Coefficients: ', regr.coef_)
#theta0
print ('Intercept: ',regr.intercept_)


#we can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


#Evaluation->we compare the actual values and predicted values to 
#calculate the accuracy of a regression model.

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

#Once there is a satisfactory model, you can use it for predictions with either existing or new data.
test_y_hat = regr.predict(test_x)


#R-squared is not error, but is a popular metric for accuracy of your model.
#It represents how close the data are to the fitted regression line.
#The higher the R-squared, the better the model fits your data.
#Best possible score is 1.0 and it can be negative
#(because the model can be arbitrarily worse).

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

























