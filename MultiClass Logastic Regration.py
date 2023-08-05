import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 22, 28, 32, 38, 48],
    'Income': [50000, 60000, 70000, 55000, 75000, 45000, 80000, 90000, 100000, 85000, 48000, 55000, 60000, 65000, 70000],
    'Education': ['High School', 'Bachelor', 'Master', 'High School', 'Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'PhD', 'Bachelor', 'High School', 'Master', 'Bachelor', 'PhD'],
    'Class': ['Class A', 'Class B', 'Class C', 'Class D', 'Class B', 'Class D', 'Class A', 'Class C', 'Class A', 'Class B', 'Class C', 'Class A', 'Class D', 'Class B', 'Class C']
}
df  = pd.DataFrame(data)
DataEncoder  = LabelEncoder() 
df['Education'] = DataEncoder.fit_transform(df['Education'])
print(df['Education']) 
df['Class'] = DataEncoder.fit_transform(df['Class'])
print(df['Class']) 
 
X_train, X_test, y_train, y_test = train_test_split(df[['Age' , 'Income' , 'Education']], df['Class'] , train_size=0.3)   
reg = LogisticRegression()  
reg.fit(X_train, y_train) 
print(X_test)
PredictedData  =  reg.predict(X_test)   
print(PredictedData) 
result    =  reg.score(X_test , y_test)
print(result) 

# plt.subplot(2, 3, 5) 
# plt =  sns.pairplot(df[['Age' , 'Income' , 'Education','Class']] , hue='Class') 
# # Show all the plots
# plt.tight_layout()
# plt.show() 
