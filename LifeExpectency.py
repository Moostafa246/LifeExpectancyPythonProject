#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:36:41 2020

@author: gurkarandhami and mustafakazim
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
raw_data = pd.read_csv("led.csv")

#generate the data we need 

country_list = []
for countries in raw_data['Country']:
    if countries not in country_list:
        country_list.append(countries)
        
year_list = []
for years in raw_data['Year']:
    if years == 2015:
        year_list.append(years)

print(len(year_list))
print(len(country_list))


#List of Columns
column = []
for cols in raw_data:
    if cols != 'Year' and cols != 'Alcohol' and cols != 'Totalexpenditure':
        column.append(cols)


Country = []
Status = []
LifeExpectancy = []
AdultMortality = []
infantdeaths = []
percentageexpenditure = []
HepatitisB = []
Measles = []
BMI = []
under_fivedeaths = []
Polio = []
Diphtheria = []
HIV_AIDS = []
GDP = []
Population = []
thinness1_19years = []
thinness5_9years = []
Incomecompositionofresources = []
Schooling = []

#since every country doesnt have data on 2015, we must ommit those countries
#from out analysis
#Populates the lists with the information from the csv file to create new df
for i in range((raw_data.shape[0])): 
    new = list(raw_data.iloc[i, :])
    if new[1] == 2015:
        Country.append(new[0])
        if new[2] == 'Developing':
            Status.append(0)
        else:
            Status.append(1)
        LifeExpectancy.append(new[3])
        AdultMortality.append(new[4])
        infantdeaths.append(new[5])
        percentageexpenditure.append(new[7])
        HepatitisB.append(new[8])
        Measles.append(new[9])
        BMI.append(new[10])
        under_fivedeaths.append(new[11])
        Polio.append(new[12])
        Diphtheria.append(new[14])
        HIV_AIDS.append(new[15])
        GDP.append(new[16])
        Population.append(new[17])
        thinness1_19years.append(new[18])
        thinness5_9years.append(new[19])
        Incomecompositionofresources.append(new[20])
        Schooling.append(new[21])

data = pd.DataFrame(list(zip(Country,
Status,
LifeExpectancy,
AdultMortality,
infantdeaths,
percentageexpenditure,
HepatitisB,
Measles,
BMI,
under_fivedeaths,
Polio,
Diphtheria,
HIV_AIDS,
GDP,
Population,
thinness1_19years,
thinness5_9years,
Incomecompositionofresources,
Schooling,)), columns = column)

#Getting rid of NA values in columns with the mean of that column
for columns in column: 
    if columns == 'Country' or columns == 'Status':
        continue
    else:
        mean = int(data[columns].mean(skipna = True))
        data[columns] = data[columns].replace(np.NaN, mean)

#plotting life expectancy, and finding the first two moments
data.hist('Lifeexpectancy')
mean_lifeexpectancy = data['Lifeexpectancy'].mean()
sd_lifeexpectancy = data['Lifeexpectancy'].std()
print("life expectancy has mean " + str(mean_lifeexpectancy) + 
      " and standard deviation " + str(sd_lifeexpectancy))

#Here I attempt to model the data using MLR with all predictors to predict
#Life expectancy 

Y = data.iloc[:, 2]

predictors = data.iloc[:, [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]


X_train, X_test, y_train, y_test = train_test_split(predictors, Y, random_state = 0, test_size = 0.2)
plt.hist(predictors)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
Y_pred = linear_regressor.predict(X_test)
#print(np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
print('Coefficients: \n', linear_regressor.coef_)
print('Mean squared error: %.2f' % mean_squared_error(y_test, Y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, Y_pred))
print(len(X_test), len(y_test))
plt.plot(X_test, Y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


