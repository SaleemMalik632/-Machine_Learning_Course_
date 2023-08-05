import pandas as pd
from sklearn.model_selection import train_test_split


data = {
    'Age': [35, 45, 28, 52, 30, 38, 29, 41, 33, 50, 36, 48, 31, 42, 27],
    'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single']
} 

df = pd.DataFrame(data)
print(df)
x = df['Age']
print(x)
y = df['Marital_Status']
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('x_train ')
print(x_train)
print('x_test')
print(x_test)
print('y_train')
print(y_train)
print('y_test ')
print(y_test)
