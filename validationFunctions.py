import pandas as pd
import numpy as np

def train_test_split(dataset: pd.DataFrame, test_ratio: float = 0.2, seed:int = 35):
    length = len(dataset)

    #shuffle the dataset
    dataset_shuffled = dataset.sample(frac=1, random_state=seed)  #'frac=1' means taking 100% of the data

    test_len = int(length*test_ratio)
    train_len = int(length*(1-test_ratio))

    training_set = dataset_shuffled.head(train_len)
    validation_set = dataset_shuffled.tail(test_len)

    return training_set, validation_set


#Kfold, takes the dataset, the number of folds and a seed and creates k_fold splits of validation/training
#the validation set length is the len of the dataset divided by the number of k_folds
def Kfold(dataset: pd.DataFrame, k_fold: int= 4, seed: int=21):
    
    length = len(dataset)

    fold_size = length // k_fold #for the cup its 1000/k for example for k= 4 it's 250
    train_list = []
    val_list = []

    #if fold is less than 2, do split train test 
    if k_fold < 2:
        raise ValueError("error: k_folds must be a value > 2")

    for fold in range(k_fold):
        #starting index and ending index for the current fold
        start_index = fold * fold_size
        #ending index is equal to the fold+1 multiplied by the fold size
        end_index = (fold + 1) * fold_size if fold < k_fold - 1 else length
        
        #current fold indexes
        test_index = range(start_index, end_index)
        #create a list of indexes that satisfy the condition. the indexes are created 
        #iterating from 0 to the length of the dataset exluding the indexes that are in test_index
        train_index = [i for i in range(length) if i not in test_index]
        
        #append the datasets to the corresponding lists
        train_list.append(dataset.iloc[train_index])
        val_list.append(dataset.iloc[test_index])

    return train_list, val_list


#split the folds in x and y for each fold
def split_folds(val_sets: list, train_sets: list, tonumpy: bool = False):
    n = len(val_sets)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    indexes_x = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',] # take the first and last 3 columns indexes
    indexes_y = ['Y1','Y2','Y3']
    if(tonumpy):
        for i in range(n):
            # print('ciclo')
            # print(i)
            # print(train_sets[i])
            x_train.append((train_sets[i].loc[:,indexes_x]).values) # Remove 3 columns
            x_test.append(val_sets[i].loc[:,indexes_x].values) 

            y_train.append(train_sets[i].loc[:,indexes_y].values)
            y_test.append(val_sets[i].loc[:,indexes_y].values)
    else:
        for i in range(n):
            # print('ciclo')
            # print(i)
            # print(train_sets[i])
            x_train.append(train_sets[i].loc[:,indexes_x]) # Remove 3 columns
            x_test.append(val_sets[i].loc[:,indexes_x]) 

            y_train.append(train_sets[i].loc[:,indexes_y])
            y_test.append(val_sets[i].loc[:,indexes_y])
    # for i in range(n):
    #     # print('ciclo')
    #     # print(i)
    #     # print(train_sets[i])
    #     x_train.append(train_sets[i].iloc[:, 0:-3]) # Remove 3 columns
    #     x_test.append(val_sets[i].iloc[:, 0:-3]) 

    #     y_train.append(train_sets[i].iloc[:,indexes])
    #     y_test.append(val_sets[i].iloc[:,indexes])

    return x_train,x_test,y_train,y_test