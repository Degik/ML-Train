import torch
import numpy as np
import pandas as pd
from torchmetrics.functional.pairwise import pairwise_euclidean_distance

def importDatasetCupInput(file_name:str, blind:bool) -> pd.DataFrame:
    dataset = []
    try:
        dataset = pd.read_csv(file_name, header=None, dtype=float)
    except Exception as e:
        print("Error | Can not read dataset cup for take input")
        exit(1)
    if not blind:
        dataset = dataset.iloc[:, :-3] # Remove 3 columns
    columns_name = ['ID'] + [f'X{i}' for i in range(1,11)]
    dataset.columns = columns_name
    dataset.set_index('ID', inplace=True)
    return dataset

def importDatasetCupOutput(file_name:str, blind:bool) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(file_name, header=None, dtype=float)
    except Exception as e:
        print("Error | Can not read dataset cup for take output")
        exit(1)
    columns_list = ['ID', 'Y1', 'Y2', 'Y3']
    if not blind: # Dataset with all inputs 
        indexes = [0, 11, 12, 13] # take the first and last 3 columns indexes
        dataset = dataset.iloc[:, indexes]
    dataset.columns = columns_list
    dataset.set_index('ID', inplace=True)
    return dataset

def importMonkDataset(file_name:str) -> pd.DataFrame:
    dataset = None
    columns_name = ["Y"] + [f"X{i}" for i in range(1,7)] + ["ID"]
    try:
        dataset = pd.read_csv(file_name, sep=" ", names=columns_name)
    except Exception as e:
        print("Error | Parsing target dataset for validation!")
        print(e)
    dataset.set_index('ID', inplace=True)
    return dataset

def takeMonkInputDataset(dataset:pd.DataFrame) -> pd.DataFrame:
    return dataset.iloc[:, 1:] #Return dataset without first and last column
 
def takeMonkOutputDataset(dataset:pd.DataFrame) -> pd.DataFrame:
    return dataset.iloc[:,[0]] #Return dataset with only first column

def importDatasetCup(file_name:str):
    try:
        dataset = pd.read_csv(file_name, header=None, dtype=float)
    except Exception as e:
        print("Error | Can not read dataset cup for take output")
        exit(1)

    columns_name = ['ID'] + [f'X{i}' for i in range(1,11)] + ['Y1','Y2','Y3'] 
    dataset.columns = columns_name
    dataset.set_index('ID', inplace=True)
    return dataset

def convert_x(x_train: np.ndarray):
    dict_3 = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    dict_2 = {1: [1, 0], 2: [0, 1]}
    dict_4 = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1]}

    new_y = []

    for row in x_train:
        new_row = []
        for j, value in enumerate(row):
            if j in [0, 1, 3]:
                new_row.extend(dict_3.get(value))
            elif j in [2, 5]:
                new_row.extend(dict_2.get(value))
            elif j == 4:
                new_row.extend(dict_4.get(value))

        new_y.append(new_row)
    return new_y


def filterElement(listToSee):
   return [elemento for elemento in listToSee if elemento <= 1.2]


def euclidean_distance_loss(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.sum(torch.square(y_pred - y_true), axis=-1)))

''' def euclidean_distance_loss(y_true, y_pred):
    return tf.norm(y_true - y_pred, ord='euclidean') '''