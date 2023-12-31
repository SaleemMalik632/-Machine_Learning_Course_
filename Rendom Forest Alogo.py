import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

Data = load_iris()
X = Data.data
Y = Data.target 
X_train, X_test, y_train, y_tes =  train_test_split(X,Y , test_size= 0.2  ) 
ObjectRandomForestClassifier   = RandomForestClassifier()
ObjectRandomForestClassifier.fit(X_train, y_train) 
print(ObjectRandomForestClassifier.predict(X_train).tolist())  
print(ObjectRandomForestClassifier.score(X_test , y_tes)) 
print(pd.Series(ObjectRandomForestClassifier.feature_importances_,index = Data.feature_names).sort_values(ascending=False) )


