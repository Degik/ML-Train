import math
import utils
import torch
import pandas as pd
import validationFunctions as vF
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class DataCup:
    def __init__(self, pathTrain:str, kfold:int = 4, seed:int = 21) -> None:
        pass
        self.k_folds = kfold
        ## TRAIN IMPORT DATASET
        dataset = utils.importDatasetCup(pathTrain)
        self.dataset_train, self.dataset_test = vF.train_test_split(dataset=dataset, seed=seed)
        # CREATE KFOLD
        self.dataset_train, self.dataset_val = vF.Kfold(self.dataset_train, self.k_folds, seed)
        #print(self.dataset_train)
        #print(self.dataset_val)
        # SPLIT FOLDS
        self.x_train, self.x_test, self.y_train, self.y_test = vF.split_folds(self.dataset_val, self.dataset_train)    
    
    def convertToTensor(self):
        ## CONVERT TO TENSOR TRAIN SET
        for i in range(self.k_folds):
            self.x_train[i] = torch.tensor(self.x_train[i].to_numpy())
            self.y_train[i] = torch.tensor(self.y_train[i].to_numpy())
            # SET TYPE DOUBLE
            self.x_train[i] = self.x_train[i].float()
            self.y_train[i] = self.y_train[i].float()
            # CONVERT TO TENSOR
            self.x_test[i] = torch.tensor(self.x_test[i].to_numpy())
            self.y_test[i] = torch.tensor(self.y_test[i].to_numpy())
            # SET TYPE DOUBLE
            self.x_test[i] = self.x_test[i].float()
            self.y_test[i] = self.y_test[i].float()
    
    def moveToGpu(self, device:str = "cuda:0"):
        for i in range(self.k_folds):
            # MOVE TENSOR TRAIN TO GPU
            self.x_train[i] = self.x_train[i].to(device)
            self.y_train[i] = self.y_train[i].to(device)
            # MOVE TENSOR TO GPU
            self.x_test[i] = self.x_test[i].to(device)
            self.y_test[i] = self.y_test[i].to(device)
        
    def createDataLoader(self, kfold) -> (DataLoader, DataLoader):
        # CREATE DATALOADER TRAIN
        batchTrain =  2
        print("Batch size for training: ", batchTrain)
        dataset_train = TensorDataset(self.x_train[kfold], self.y_train[kfold])
        data_loader_train = DataLoader(dataset_train, batch_size=batchTrain, shuffle=False)
        # CREATE DATALOADER TEST
        batchTest = 2
        print("Batch size for testing: ", batchTest)
        dataset_test = TensorDataset(self.x_test[kfold], self.y_test[kfold])
        data_loader_test = DataLoader(dataset_test, batch_size=batchTest, shuffle=False)
        return data_loader_train, data_loader_test
