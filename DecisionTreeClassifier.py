import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree


data = {
    'Age': [35, 45, 28, 52, 30, 38, 29, 41, 33, 50, 36, 48, 31, 42, 27],
    'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single']
}


df = pd.DataFrame(data)
Marital_Status = LabelEncoder()
df['Marital_Status'].replace({'Single': 0, 'Married': 1}, inplace=True)
print(df['Marital_Status'])
X_train, X_test, y_train, y_test = train_test_split(
    df[['Age']], df['Marital_Status'], test_size=0.2)
classfier = DecisionTreeClassifier()
classfier.fit(X_train, y_train) 
print(X_test)
print(classfier.predict(X_test))
print(classfier.score(X_test, y_test))
print(tree.plot_tree(classfier)) 
