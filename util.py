import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def get_processing_parameters(train_data: pd.DataFrame) -> dict:
    """
    Obtain parameters used to process the training data
    so same parameters can be used to transform the test data
    """

    process_parameters = {}
    
    # obtain median values by Pclass and Sex group
    # https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial/notebook#1.2.1-Age
    grouped_pclass_medians = train_data.groupby(["Pclass", 'Sex'])["Age"].median()
    for key in grouped_pclass_medians.keys():
        process_parameters[f"Pclass_{key[0]}_{key[1]}_median_age"] = grouped_pclass_medians.get(key)

    # using MinMaxScaler correctly:
    # https://stackoverflow.com/questions/50565937/how-to-normalize-the-train-and-test-data-using-minmaxscaler-sklearn
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data[['Age', 'Fare']])
    # saving scaler model for later
    # https://stackoverflow.com/questions/41993565/save-minmaxscaler-model-in-sklearn
    joblib.dump(min_max_scaler, 'data/min_max_scaler.gz')
    
    return process_parameters

def process_data(data: pd.DataFrame, process_parameters: dict) -> np.array:
    """
    Process training and testing data with the same parameters 
    used to transform the training data
    """

    # fill missing 'Age' attribute with median value from class and sex group individual belogs to
    for index, row in data.iterrows():
        if pd.isnull(row.Age):
            data.at[index, "Age"] = process_parameters[f"Pclass_{row.Pclass}_{row.Sex}_median_age"]

    # replace names with titles
    data["Name"] = data["Name"].apply(lambda x: x.split(", ")[1].split(". ")[0])

    # scale data
    min_max_scaler = joblib.load('data/min_max_scaler.gz')
    # https://www.kaggle.com/code/rtatman/data-cleaning-challenge-scale-and-normalize-data/notebook
    data[['Age', 'Fare']] = min_max_scaler.transform(data[['Age', 'Fare']])
    
    # we don't care about 'PassengerId' for training
    # 'Tickets' contains a mix of integers with strings
    # 'Cabin' contains large amounts of missing data, which I don't know how to handle rn
    # don't know how to process Name (titles) rn
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    data_encoded = pd.get_dummies(data)

    return data_encoded.to_numpy()
