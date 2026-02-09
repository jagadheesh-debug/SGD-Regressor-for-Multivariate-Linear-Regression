# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.start 2. Importing necessary liberaries 3. Data preprocessing 4. Spliting data int training and
testing data 5. Performing SGD-Regressor 6. Calculating error 7. end

## Program:
```
/*
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
# Sample data (2 features)
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])
# Create model
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')
# Train model
model.fit(X, y)
# Check learned weights
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
# Predict
y_pred = model.predict(X)
# Plot Actual vs Predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') # Perfect prediction
line
plt.show()
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:jagadheesh kumar T 
RegisterNumber:212225040139  
*/
```

## Output:
<img width="691" height="615" alt="image" src="https://github.com/user-attachments/assets/72b58b89-bb45-4a0e-a06e-3e92d80c2d8c" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
