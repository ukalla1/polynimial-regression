import pandas as pd 									
import numpy as np 										
from sklearn.linear_model import LinearRegression			# sklearn library for linear regression
from sklearn.preprocessing import PolynomialFeatures        # --""--""--""--""--""-polynomial feature conversion
import matplotlib.pyplot as plt 							# matplot lib for visualization

# Read the csv file and store in a var
data = pd.read_csv('./Position_Salaries.csv')

x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]

#split the data into train and test sets
split_len = int(len(x) * 0.8)
x_train, x_test = data.iloc[:split_len, 1:2].values, data.iloc[split_len:, 1:2].values
y_train, y_test = data.iloc[:split_len, 2:].values, data.iloc[split_len:, 2:].values

# get polynomial features
polynomial_features_train = PolynomialFeatures(degree = 5)
polynomial_features_test = PolynomialFeatures(degree = 5)
x_train_poly = polynomial_features_train.fit_transform(x_train)
x_test_poly = polynomial_features_test.fit_transform(x_test)

# create the polynimoal regressor
polynomial_regressor = LinearRegression()
polynomial_regressor.fit(x_train_poly, y_train)

# predict on the test set
y_pred = polynomial_regressor.predict(polynomial_features_test.fit_transform(x_test))

# visualize the test and the train sets fit for correctness
x_grid_train = np.arange(min(x_train), max(x_train), 0.1)
x_grid_train = x_grid_train.reshape((len(x_grid_train), 1))
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_grid_train, polynomial_regressor.predict(polynomial_features_train.fit_transform(x_grid_train)), color = 'blue')
plt.title('polynimial regressor training data plot')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

x_grid_test = np.arange(min(x_test), max(x_test), 0.1)
x_grid_test = x_grid_test.reshape((len(x_grid_test), 1))
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_grid_test, polynomial_regressor.predict(polynomial_features_train.fit_transform(x_grid_test)), color = 'blue')
plt.title('polynimial regressor training data plot')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

# sample prediction
print(polynomial_regressor.predict(polynomial_features_test.fit_transform([[6.5]])))