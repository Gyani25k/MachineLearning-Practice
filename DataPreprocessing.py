import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data=pd.read_csv(r'Data.csv')
df=pd.DataFrame(data)

'''Alternate Method 
X=df.drop(columns=['Purchased'])
y=df['Purchased]'''

X=data.iloc[ : ,:-1].values 
y=data.iloc[ : , -1].values

print(X)
print(y)

# Handaling Missing Values 

from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan,strategy="mean")
X[:,1:3] = impute.fit_transform(X[:,1:3])

# Encoding Categorical Data Using OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

# Encoding Categorical Data Using LabelEncoder
from sklearn.preprocessing import LabelEncoder
lt=LabelEncoder()
y=lt.fit_transform(y)

# Spliting Into Training and Testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

# Feture Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[: ,3:]=sc.fit_transform(X_train[: ,3:])
X_test[: ,3:]=sc.fit_transform(X_test[: ,3:])


print(X_train)
print(y_train)

