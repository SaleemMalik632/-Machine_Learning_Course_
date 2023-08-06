import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset for three languages: English, French, and Spanish
texts = [
    'This is an example sentence in English.',
    'Voici une phrase exemple en français.',
    'Esta es una oración de ejemplo en español.'
]

labels = ['English', 'French', 'Spanish'] 

# Creating feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Multiclass Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
