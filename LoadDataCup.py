import math
import utils
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class DataCup:
    def __init__(self, pathTrain:str) -> None:
        pass   
        ## TRAIN IMPORT DATASET
        self.x_train = utils.importDatasetCupInput(pathTrain, blind=False)
        self.y_train = utils.importDatasetCupOutput(pathTrain, blind=False)
    
    def splitData(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=0.2,random_state=39)    
    
    def convertToTensor(self):
        ## CONVERT TO TENSOR TRAIN SET
        self.x_train = torch.tensor(self.x_train.to_numpy())
        self.y_train = torch.tensor(self.y_train.to_numpy())
        # SET TYPE DOUBLE
        self.x_train = self.x_train.float()
        self.y_train = self.y_train.float()
        # CONVERT TO TENSOR
        self.x_test = torch.tensor(self.x_test.to_numpy())
        self.y_test = torch.tensor(self.y_test.to_numpy())
        # SET TYPE DOUBLE
        self.x_test = self.x_test.float()
        self.y_test = self.y_test.float()
    
    def moveToGpu(self, device:str = "cuda:0"):
        # MOVE TENSOR TRAIN TO GPU
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)
        # MOVE TENSOR TO GPU
        self.x_test = self.x_test.to(device)
        self.y_test = self.y_test.to(device)
        
    def createDataLoader(self) -> (DataLoader, DataLoader):
        # CREATE DATALOADER TRAIN
        batchTrain =  64
        print("Batch size for training: ", batchTrain)
        dataset_train = TensorDataset(self.x_train, self.y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=batchTrain, shuffle=True)
        # CREATE DATALOADER TEST
        batchTest = 64
        print("Batch size for testing: ", batchTest)
        dataset_test = TensorDataset(self.x_test, self.y_test)
        data_loader_test = DataLoader(dataset_test, batch_size=batchTest, shuffle=True)
        return data_loader_train, data_loader_test