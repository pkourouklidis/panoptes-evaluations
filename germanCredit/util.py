import numpy as np
import pandas as pd
import requests

def generateFeatures(df):
    
    #Feature engineering
    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)

    df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
    df['Checking account'] = df['Checking account'].fillna('no_inf')

    #Purpose to Dummies Variable
    df = df.merge(pd.get_dummies(df.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
    #Sex feature in dummies
    df = df.merge(pd.get_dummies(df.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    # Housing get dummies
    df = df.merge(pd.get_dummies(df.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    # Housing get Saving Accounts
    df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    # Housing get Risk
    df = df.merge(pd.get_dummies(df.Risk, prefix='Risk'), left_index=True, right_index=True)
    # Housing get Checking Account
    df = df.merge(pd.get_dummies(df["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    # Housing get Age categorical
    df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

    #Excluding the missing columns
    del df["Saving accounts"]
    del df["Checking account"]
    del df["Purpose"]
    del df["Sex"]
    del df["Housing"]
    del df["Age_cat"]
    del df["Risk"]
    del df['Risk_good']

    df['Credit amount'] = np.log(df['Credit amount'])
    X = df.drop('Risk_bad', axis = 1)
    y = df["Risk_bad"]
    return X,y

def sendInferenceRequest(input):
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
            "http://credit-predictor-default.evaluations.panoptes.uk/v2/models/credit/infer",
            json = body,
            headers = {"accept-encoding": None}
        )

    return response.json()['outputs'][0]['data'][0]