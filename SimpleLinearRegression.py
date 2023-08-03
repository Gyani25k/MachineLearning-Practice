import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'Salary_Data.csv')
df=pd.DataFrame(data)

print(df)

# plt.plot(df['YearsExperience'],df['Salary'],'--')

# plt.show()