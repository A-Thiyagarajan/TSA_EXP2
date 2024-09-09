#### Developed by Thiyagarajan A
#### Register no: 212222240110
#### Date:
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Load and prepare data: Load the stock data, convert 'Date' to datetime, and reset the index for model fitting.

Extract features and target: Use a numerical index (X) as the feature and 'Open' prices (y) as the target.

Fit models: Fit a linear regression model and a polynomial regression model (degree 2) to the data.

Predict trends: Generate predictions for both linear and polynomial trends.

Visualize and display: Plot the actual data, linear trend, and polynomial trend, and print the trend equations.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data from the CSV file
file_path = 'Google_Stock_Price_Train.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Use only numerical index for fitting models
data.reset_index(inplace=True)

# Extract the features and target
X = np.arange(len(data)).reshape(-1, 1)  # Days as feature
y = data['Open'].values  # Open prices as the target


# Linear Trend
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Plotting the actual data and linear trend
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], y, label='Actual Data', color='blue')
plt.plot(data['Date'], y_linear_pred, label='Linear Trend', linestyle='--', color='green')
plt.title('Google Stock Price with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# Print the linear trend equation
print(f"Linear Trend Equation: y = {linear_model.coef_[0]:.2f} * x + {linear_model.intercept_:.2f}")


# Polynomial Trend (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# Plotting the actual data and polynomial trend
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], y, label='Actual Data', color='blue')
plt.plot(data['Date'], y_poly_pred, label='Polynomial Trend (Degree 2)', linestyle='-.', color='red')
plt.title('Google Stock Price with Polynomial Trend (Degree 2)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# Print the polynomial trend equation
print("Polynomial Trend Equation (Degree 2): y = {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    poly_model.coef_[2], poly_model.coef_[1], poly_model.intercept_))

```
### OUTPUT
![image](https://github.com/user-attachments/assets/33e57543-bde2-4f3c-bfb1-a1e13e1502e8)

![image](https://github.com/user-attachments/assets/12c8c4ba-bea5-4410-8a77-33dd09828e59)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
