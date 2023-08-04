import pandas as pd
from sklearn import linear_model
import numpy as np

data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Loan': [5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000]
}
df = pd.DataFrame(data)
reg = linear_model.LinearRegression()
reg.fit(df['Age'].values.reshape(-1, 1), df['Loan'])
print('inter cept of the graph ')
print(reg.intercept_)
print('coef of the graph')
print(reg.coef_)
datanew = reg.predict([[90]])
print(datanew)

# learn  regration with multiple variable
data_ = {
    'Age': [25, 30, np.nan, 40, 45, 50, 55, np.nan, 65, 70],
    'Weight': [70, 75, 80, 85, np.nan, 95, 100, 105, 110, 115],
    'Height': [np.nan, 165, 170, 175, 180, 185, 190, 195, 200, 205],
    'Loan': [5000, 7000, 9000, np.nan, 13000, 15000, 17000, 19000, np.nan, 23000],
}
df = pd.DataFrame(data_)
df.fillna(df.mean(), inplace=True)  # Fill missing values with column means
print(df)
reg2 = linear_model.LinearRegression()
print('Prediction of the multiple variable Dataset')
reg2.fit(df[['Age', 'Weight', 'Height']].values.reshape(-1, 3), df['Loan'])
datanew = reg2.predict([[90, 100, 180]])
print(datanew)
