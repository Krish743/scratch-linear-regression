import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linearregression import LinearRegression


X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise =20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=123)

reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

def mse(y_test, pred):
    return np.mean((y_test - pred)**2)

mse = mse(y_test, pred)
print(mse)


pred_line = reg.predict(X)
plt.scatter(X, y, color='c', label='Actual data')    
plt.plot(X, pred_line, color='r', label='Regression line') 
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()