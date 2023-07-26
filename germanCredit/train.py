import copy
import random
from pickle import dump

import numpy as np  # Math library
import pandas as pd  # To work with dataset
from sklearn.metrics import classification_report  # To evaluate our model
from sklearn.model_selection import train_test_split  # to split the data
from sklearn.naive_bayes import GaussianNB
from util import generateFeatures

#Importing the data
random.seed(42)
df_credit = pd.read_csv("./dataset/german_credit_data.csv",index_col=0)

X,y = generateFeatures(copy.deepcopy(df_credit))
print(X)
# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
GNB = GaussianNB()
model = GNB.fit(X_train, y_train)

# Printing the Training Score
print("Training data accuracy:")
print(model.score(X_train, y_train))

y_pred = model.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))

with open("model.joblib", "wb") as file:
    dump(model, file)