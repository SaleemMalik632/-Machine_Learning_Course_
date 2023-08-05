import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from  sklearn.datasets import load_iris

data  = load_iris()
x= data.data
y = data.target

X_train, X_test, y_train, y_test =   train_test_split(x, y , train_size=0.2)

Modal = SVC(kernel='linear') 
Modal.fit(X_train, y_train)
print(Modal.predict(X_test)) 

print(Modal.score(X_test , y_test) ) 

