#Load the libraries
import pandas as pd #To work with dataset
import numpy as np #Math library
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model
from pickle import dump,load
import requests

#Importing the data
df_credit = pd.read_csv("./dataset/german_credit_data.csv",index_col=0)
# print(df_credit)

#Feature engineering
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)

df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

#Purpose to Dummies Variable
df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#Sex feature in dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
# Housing get dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
# Housing get Saving Accounts
df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
# Housing get Risk
df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
# Housing get Checking Account
df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
# Housing get Age categorical
df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

#Excluding the missing columns
del df_credit["Saving accounts"]
del df_credit["Checking account"]
del df_credit["Purpose"]
del df_credit["Sex"]
del df_credit["Housing"]
del df_credit["Age_cat"]
del df_credit["Risk"]
del df_credit['Risk_good']
del df_credit["Risk_bad"]

df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
print(*[c for c in df_credit.columns])
input = df_credit.loc[0].tolist()
print(input)
body = {
        "id": "1",
        "inputs": [
            {
                "name": "input",
                "shape": [1,24],
                "datatype": "FP32",
               "data": input
            }
        ],
        "outputs": [{"name": "predict"}]
    }
response = requests.post(
        "http://credit-scorer-predictor-default.kubeflow-user-example-com.panoptes.uk/v2/models/credit-scorer/infer",
        json = body,
        headers = {"accept-encoding": None}
    )

print(response.json()['outputs'][0]['data'][0])