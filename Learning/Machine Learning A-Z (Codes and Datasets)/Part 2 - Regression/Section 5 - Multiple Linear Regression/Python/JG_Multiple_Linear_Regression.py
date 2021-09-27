import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("Original Table")
print(X)
print('\n')

# Encoding categorical data

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("Table after applying hot encoding")
print(X)
print('\n')

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print("The Predicted profits and the real profits")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
print('\n')
