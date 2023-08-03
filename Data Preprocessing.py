from json import encoder
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the data from 'Churn_Modelling.csv'
data = pd.read_csv('Churn_Modelling.csv')


# Step 2: Select the columns you want to impute (assuming you want to impute numerical columns)
numerical_columns = ['CreditScore']
# Step 3: Handle missing values in numerical columns using SimpleImputer with 'mean' strategy
imputer = SimpleImputer(strategy='mean')
newdata= imputer.fit_transform(data[numerical_columns])

# Sample categorical data
categories = ['cat', 'dog', 'bird', 'cat', 'dog']
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories) 
print("Original Data:", categories)
print("Encoded Data:", encoded_categories)

print('Original Data:')
print(data)




# for the onhot encoder 

categories_ = [['cat'], ['dog'], ['bird'], ['cat'], ['dog']]

encoder_ = OneHotEncoder()


encoder_Data = encoder_.fit_transform(categories_)
encoded_array = encoder_Data.toarray()
print('this is the hotencode data ') 
print(encoded_array)



