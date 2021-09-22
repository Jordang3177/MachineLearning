# Simple Linear Regression

# Importing the libraries

import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset in to the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Simple Linear Regression model on the Training Set

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test Set results

y_pred = regressor.predict(x_test)

# Visualising the Training Set results


