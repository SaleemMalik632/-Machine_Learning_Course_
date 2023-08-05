import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'Age': [25, 30,35, 40, 45, 50, 55, 55, 65, 70 ,75, 80 , 85 , 90 , 95 ],
    'insurance': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',    'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes'],
}

df = pd.DataFrame(data)

df['insurance'].replace({'No':0, 'Yes': 1}, inplace=True)
# print(df)
X_train, x_test, y_train, y_test = train_test_split( df[['Age']], df['insurance'], test_size=0.3)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
pre  =  LogReg.predict(x_test)
print(pre)
pre  = LogReg.predict([[100]])
print(pre) 
 