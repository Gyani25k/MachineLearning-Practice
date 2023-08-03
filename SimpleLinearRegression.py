import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'Salary_Data.csv')
df = pd.DataFrame(data)

# Checkin for Columns of DataFrame
print(df.columns)

# Check for Missing Values
print(df.isnull().sum())

# Dividing into 2 Datasets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Dividing into Training and Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("X Train Shape", X_train.shape)
print("X Test Shape", X_test.shape)
print("Y Train Shape", y_train.shape)
print("Y Test Shape", y_test.shape)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print("Model Training Started")
regressor.fit(X_train, y_train)
print("Model Trained Successfully")

predict = regressor.predict(X_test)
print(predict)

plt.scatter(X_train.ravel(), y_train, color='Red')
plt.plot(X_train, regressor.predict(X_train), color='Blue')  
plt.title("Simple Regression Model")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()
