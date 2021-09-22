import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original X")
print(x)
print('\n')
print("Original Y")
print(y)
print('\n')

# Taking care of Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print("X after taking care of the Missing Data")
print(x)
print('\n')

# Encoding Categorical data

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print("X after encoding the Independent Variable")
print(x)
print('\n')

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print("Y after encoding the Dependent Variable")
print(y)
print('\n')

# Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("Training Data from X")
print(x_train)
print('\n')

print("Test Data from X")
print(x_test)
print('\n')

print("Training Data from Y")
print(y_train)
print('\n')

print("Test Data from Y")
print(y_test)
print('\n')

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print("Training Data from X after doing Feature Scaling")
print(x_train)
print('\n')

print("Test Data from X after doing Feature Scaling")
print(x_test)
print('\n')