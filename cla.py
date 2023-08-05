import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = {
    'Age': [35, 45, 28, 52, 30, 38, 29, 41, 33, 50, 36, 48, 31, 42, 27],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Income': [50000, 60000, 42000, 80000, 55000, 75000, 48000, 70000, 52000, 90000, 58000, 82000, 49000, 72000, 43000],
    'GET': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
df['Gender'].replace({'Male':1 ,'Female':0 } , inplace= True) 

x_train , x_test  , y_train ,y_test =  train_test_split(df[['Age' , 'Gender' , 'Income']] , df["GET"] , test_size= 0.2 )

print(x_test)

Modal = GaussianNB()
Modal.fit(x_train , y_train) 
PredictedData  =  Modal.predict(x_test)   
print(PredictedData) 
result    =  Modal.score(x_test , y_test)
print(result) 