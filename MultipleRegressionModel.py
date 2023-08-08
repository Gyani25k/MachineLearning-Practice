import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset

data=pd.read_csv(r'50_Startups.csv')
dataset=pd.DataFrame(data)


print("Checking Of Null Values in Dataset ::")
print(dataset.isnull().sum())

print("Describing Dataset ::")
print(dataset.describe())


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# Dealing With Categorical Data 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("X Train Dataset Shape ::",x_train.shape)
print("X Test Dataset Shape ::",x_test.shape)
print("Y Train Dataset Shape ::",y_train.shape)
print("Y Test Dataset Shape ::",y_test.shape)

