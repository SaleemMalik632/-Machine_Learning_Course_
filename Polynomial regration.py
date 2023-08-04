import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data = {
    'Employee': ['John', 'Alice', 'Bob', 'Emily', 'Michael', 'Sarah', 'David', 'Olivia', 'James', 'Emma'],
    'Position': ['Manager', 'Assistant', 'Associate', 'Associate', 'Assistant', 'Manager', 'Associate', 'Assistant', 'Manager', 'Associate'],
    'Level': [3, 2, 1, 2, 1, 3, 2, 1, 3, 1],
    'Salary': [80000, 60000, 45000, 65000, 55000, 75000, 48000, 70000, 85000, 50000]
}

df = pd.DataFrame(data) 
x = df[['Level']].values 
y = df['Salary'] 
print(x) 
print(y) 

reg = linear_model.LinearRegression() 
reg.fit(df['Level'].values.reshape(-1, 1),y)  
datanew = reg.predict([[2.5]])
print(datanew)
ploy = PolynomialFeatures(degree=2) 
x_poly = ploy.fit_transform(x) 
reg2 = linear_model.LinearRegression() 
reg2.fit(x_poly , y)
newdata = reg2.predict(ploy.fit_transform([[2.5]]))
print(newdata)  