import numpy
import pandas
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

data = {
    'Age': [35, 45, 28, 52, 30, 38, 29, 41, 33, 50, 36, 48, 31, 42, 27],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Income': [50000, 60000, 42000, 80000, 55000, 75000, 48000, 70000, 52000, 90000, 58000, 82000, 49000, 72000, 43000],
    'Education': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor', 'Master', 'High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'Bachelor', 'Master'],
    'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single']
}
# lable encoder is only apply on the singe colume of the data
labelencodedata = LabelEncoder()
gender = data['Gender']
encodedata = labelencodedata.fit_transform(gender)
print('label encoded data')
print(encodedata)

data_ = [
    [35, 'Male', 50000, 'Bachelor', 'Single'],
    [45, 'Female', 60000, 'Master', 'Married'],
    [28, 'Male', 42000, 'High School', 'Single'],
    [52, 'Male', 80000, 'PhD', 'Married'],
    [30, 'Female', 55000, 'Bachelor', 'Single'],
    [38, 'Female', 75000, 'Master', 'Married'],
    [29, 'Male', 48000, 'High School', 'Single'], 
    [41, 'Male', 70000, 'Bachelor', 'Married'],
    [33, 'Female', 52000, 'Master', 'Single'],
    [50, 'Female', 90000, 'PhD', 'Married'],
    [36, 'Male', 58000, 'Bachelor', 'Single'],
    [48, 'Female', 82000, 'Master', 'Married'],
    [31, 'Male', 49000, 'High School', 'Single'],
    [42, 'Male', 72000, 'Bachelor', 'Married'],
    [27, 'Female', 43000, 'Master', 'Single']
]

hotencoded = OneHotEncoder()
hotdata = hotencoded.fit_transform(data_)
arraydata = hotdata.toarray()

print('Hot Encoded Data')

print(arraydata)


# train_test_split this is used to testing the model on the import data 

X_train, X_test, y_train, y_test = train_test_split(data, data_, test_size=0.3, random_state=42)

print(X_train) 
print(X_test)
print(y_train) 
print(y_test)
 




