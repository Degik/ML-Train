import math
import utils
import torch
from torch.utils.data import TensorDataset, DataLoader

class DataMonk:
    def __init__(self, pathTrain:str, pathTest:str) -> None:
        pass   
        ## TRAIN IMPORT DATASET
        self.dataset_train = utils.importMonkDataset(pathTrain)
        self.x_train = utils.takeMonkInputDataset(self.dataset_train)
        self.y_train = utils.takeMonkOutputDataset(self.dataset_train)
        # CONVERT DATA FOR TRAINING
        self.x_train = utils.convert_x(self.x_train.to_numpy())
        self.y_train = self.y_train.to_numpy()
        ## TEST IMPORT DATASET
        self.dataset_test = utils.importMonkDataset(pathTest)
        self.x_test = utils.takeMonkInputDataset(self.dataset_test)
        self.y_test = utils.takeMonkOutputDataset(self.dataset_test)
        # CONVERT DATA FOR TESTING
        self.x_test = utils.convert_x(self.x_test.to_numpy())
        self.y_test = self.y_test.to_numpy()
        
    def convertToTensor(self):
        ## CONVERT TO TENSOR TRAIN SET
        self.x_train = torch.tensor(self.x_train)
        self.y_train = torch.tensor(self.y_train)
        # SET TYPE DOUBLE
        self.x_train = self.x_train.double()
        self.y_train = self.y_train.double()
        # CONVERT TO TENSOR
        self.x_test = torch.tensor(self.x_test)
        self.y_test = torch.tensor(self.y_test)
        # SET TYPE DOUBLE
        self.x_test = self.x_test.double()
        self.y_test = self.y_test.double()
    
    def moveToGpu(self):
        # MOVE TENSOR TRAIN TO GPU
        self.x_train = self.x_train.to("cuda:0")
        self.y_train = self.y_train.to("cuda:0")
        # MOVE TENSOR TO GPU
        self.x_test = self.x_test.to("cuda:0")
        self.y_test = self.y_test.to("cuda:0")
        
    def createDataLoader(self) -> (DataLoader, DataLoader):
        # CREATE DATALOADER TRAIN
        size = self.x_train.size(0)
        batchTrain =  math.ceil(size/2)
        print("Batch size for training: ", batchTrain)
        dataset_train = TensorDataset(self.x_train, self.y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=batchTrain, shuffle=True)
        # CREATE DATALOADER TEST
        size = self.x_test.size(0)
        batchTest = math.ceil(size/2)
        print("Batch size for testing: ", batchTest)
        dataset_test = TensorDataset(self.x_test, self.y_test)
        data_loader_test = DataLoader(dataset_test, batch_size=batchTest, shuffle=True)
        return data_loader_train, data_loader_test
