import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score 



# Given dataset
data = {
    'Age': [35, 45, 28, 52, 30, 38, 29, 41, 33, 50, 36, 48, 31, 42, 27],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Income': [50000, 60000, 42000, 80000, 55000, 75000, 48000, 70000, 52000, 90000, 58000, 82000, 49000, 72000, 43000],
    'Education': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor', 'Master', 'High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'Bachelor', 'Master'],
    'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single'],
    'Target_Variable': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)
print(df) 

# Encode categorical columns 'Gender' and 'Marital_Status' using LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])

# Split the data into features (X) and target (y) variables
X = df.drop(columns='Target_Variable')
y = df['Target_Variable']

# Apply OneHotEncoder to the 'Education' column
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = onehot_encoder.fit_transform(X[['Education']])
X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(['Education']))
X = pd.concat([X.drop(columns='Education'), X_encoded_df], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("1):Linear Regression Predictions:")
print(lr_predictions)

# 2. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
print("2)Logistic Regression Predictions:")
print(logistic_predictions)

# 3. Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("3):Decision Tree Predictions:")
print(dt_predictions)

# 4. Support Vector Machine (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print("4):SVM Predictions:")
print(svm_predictions)

# 5. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
print("5):Naive Bayes Predictions:")
print(nb_predictions)


# 6  K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train) 
knn_predictions = knn_model.predict(X_test)
# Evaluate the KNN model
accuracy = accuracy_score(y_test, knn_predictions)
print("6): KNN Accuracy:", accuracy)

# this is :AdaBoost Predictions: 
adaboost_model = AdaBoostClassifier()
adaboost_model.fit(X_train, y_train)
adaboost_predictions = adaboost_model.predict(X_test)
print("7):AdaBoost Predictions:")
print(adaboost_predictions)


# 8. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("8):Random Forest Predictions:")
print(rf_predictions)

# 9. Dimensionality Reduction - PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("9):PCA Reduced Data:")
print(X_pca)

# 10. Gradient Boosting and AdaBoost
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print("10):Gradient Boosting Predictions:")
print(gb_predictions)

