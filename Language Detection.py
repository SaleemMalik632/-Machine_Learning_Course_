import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('Language Detection.csv')

x = data['Text']
Label = data['Language']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(X, Label)    

Model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
Model.fit(X, Label) 
text= 'жаль что ты со школы так изменилась, пьешь много?'
TestSimple = vectorizer.transform([text]) 
print(Model.score(X_test , y_test)) 
TestSimple_pred = Model.predict(TestSimple)
print(TestSimple_pred)
