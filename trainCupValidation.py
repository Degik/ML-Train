import os
import torch
import NetCup
import numpy as np
import LoadDataCupValidation
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import validationFunctions as vF
import IPython.display as display

from itertools import product



print("TRAINING CUP DATASET")
# PATH
pathTrain = "CUP/ML-CUP23-TRAIN.csv"
pathTestInput = "CUP/ML-CUP23-TEST-INPUT.csv"
pathTestTarget = "CUP/ML-CUP23-TEST-TARGET.csv"
seed = 15
# HYPERPARAMETER
num_epochs = 2000
#momentum = 0.9
threshold = 0.01
#penality = 0.0005

#grid search
layers_conf = [[10, 256, 256, 300, 3], [10, 40, 40, 80, 3], [10, 64, 128, 200, 128, 3]]
#layers_conf = [[10, 256, 256, 300, 3]]
activation_functions = ['tanh']
optimizers = ['sgd']
penalities = [0.001, 0.0005, 0.0001]
#penalities = [0.001]
momentums = [0.9]
learning_rates = [0.001, 0.0008, 0.0005]
#learning_rates = [0.001]
#
k_folds = 4
#
numberTest = len(layers_conf) * len(activation_functions) * len(optimizers) * len(penalities) * len(momentums) * len(learning_rates)
bestResults = []

# IMPORT DATA
dataCup = LoadDataCupValidation.DataCup(pathTrain, k_folds, seed)
# DATA: TENSOR, GPU, DATALOADER
dataCup.convertToTensor()
# MOVE TO GPU
device = "cuda:0"
dataCup.moveToGpu(device=device)


for number, config in enumerate(product(layers_conf, activation_functions, optimizers, penalities, momentums, learning_rates)):
    layers, activation, optimizerName, penality, momentum, lr = config
    # PATH
    testName = f"{layers}-{optimizerName}-{activation}-{penality}-{momentum}-{lr}"
    pathName = f'modelsCup/Cup-{testName}'
    bestResults.append(testName + "\n")

    # CREATE DIR
    os.makedirs(pathName, exist_ok=True)
    
    history_train = []
    history_val = []
    
    for kfold in range(k_folds):
        # CREATE NET
        structureNet = []
        # If you need to change the neurons number go to netCup.py
        print("Load regressor [net]")
        net = NetCup.NetCupRegressor(layers, structureNet, activation)
        # MOVE NET TO GPU
        net = net.to(device)
        # SET TYPE NET
        net = net.float()
        # OPTIMIZER AND CRITERION
        # MSELoss for Regressor
        # SGD for Regressor
        print("Load MSELoss [criterion]\nLoad SGD [optimizer]")
        criterion = nn.MSELoss()
        optimizer = None
        if optimizerName == "adam":
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=penality)
        elif optimizerName == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=penality)
        else:
            print("OPTIMIZER NON TROVATO!")
            exit(1)
        
        data_loader_train, data_loader_val = dataCup.createDataLoader(kfold)
        #Values used for graphs
        loss_values_train = []
        accuracy_values_train = []
        loss_values_val = []
        accuracy_values_val = []
        # BEST
        best_accuracy_train = 0.0
        best_accuracy_val = 0.0
        best_loss_train = 100.0
        best_loss_val = 100.0
        #
        results = []
        net.train()
        for epoch in range(num_epochs):
            total_loss = 0
            total = 0
            correct = 0

            for batch_input, batch_output in data_loader_train:
                #Forward pass
                outputs = net(batch_input)
                #Training loss
                loss = criterion(outputs, batch_output)
                #Calculate total loss
                total_loss += loss.item()
                #Backward and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss_train = total_loss / len(data_loader_train)
            #Add to list
            loss_values_train.append(avg_loss_train)

            total = 0
            correct = 0
            #CALCULATE ACCURACY VAL
            net.eval()
            total_loss = 0
            with torch.no_grad():
                for batch_input, batch_output in data_loader_val:
                    outputs = net(batch_input)
                    loss = criterion(outputs, batch_output)
                    total_loss += loss.item()
                avg_loss_val = total_loss / len(data_loader_val)
                loss_values_val.append(avg_loss_val)
            net.train()
            
            result = f'KFold[{kfold+1}/{k_folds}] --> Epoch[{epoch+1}/{num_epochs}] Learning-rate: {lr}, Loss-Train: {avg_loss_train:.4f}, Loss-Val: {avg_loss_val:.4f}'
            print(f"GridSearch[{number+1}/{numberTest}] --> " + result)
            
            #Set best loss
            best_loss_train = min(best_loss_train, avg_loss_train)
            best_loss_val = min(best_loss_val, avg_loss_val)
            #List append
            results.append(result) 

        #END EPOCHS
        
        history_train.append(loss_values_train)
        history_val.append(loss_values_val)
        
        #Save best results
        bestPrint = f'     KFold-{kfold} -> Best-loss-train: {best_loss_train:.4f}, Best-loss-val: {best_loss_val:.4f} \n'
        bestResults.append(bestPrint)
        
        with open(f"{pathName}/results_kfold-{kfold}.txt", 'w') as file:
            for res in results:
                file.write(res + "\n")
    #END KFOLD
    
    mean_train_loss = np.mean(history_train, axis=0)
    mean_val_loss = np.mean(history_val, axis=0)
    #Last
    last_train_loss = [lst[-1] for lst in history_train]
    last_val_loss = [lst[-1] for lst in history_val]
    #Mean last
    mean_last_train_loss = np.mean(last_train_loss)
    mean_last_val_loss = np.mean(last_val_loss)
    #Adding best results
    bestPrint = f"     Mean-Last-Epoch-Train: {mean_last_train_loss:.4f}, Mean-Last-Epoch-Val: {mean_last_val_loss:.4f}\n"
    bestResults.append(bestPrint)
    
    #Save plot loss
    display.clear_output(wait=True)
    plt.plot(mean_train_loss, label='Training Loss')
    plt.plot(mean_val_loss, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Mean-Loss per Epoch')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.savefig(f'{pathName}/Mean-Loss.png')
    plt.clf()
    
    #Save plot loss for training kfold
    display.clear_output(wait=True)
    for testNumber in range(k_folds):
        plt.plot(history_train[testNumber], label=f'Train-Loss-{testNumber}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Mean-Loss per Epoch')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.savefig(f'{pathName}/KFold-Loss-Train.png')
    plt.clf()
    
    #Save plot loss for validation kfold
    display.clear_output(wait=True)
    display.clear_output(wait=True)
    for testNumber in range(k_folds):
        plt.plot(history_val[testNumber], label=f'Val-Loss-{testNumber}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Mean-Loss per Epoch')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.savefig(f'{pathName}/KFold-Loss-Val.png')
    plt.clf()

    #Save plot accuracy
    ''' display.clear_output(wait=True)
    plt.plot(accuracy_values_train, label='Accuracy Train')
    plt.plot(accuracy_values_val, label='Accuracy Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Epoch kfold-{kfold}')
    plt.legend()
    plt.savefig(f'{pathName}/Accuracy-test.png')
    plt.clf() '''
    
    #Save model
    torch.save(net, f'{pathName}/model.pth')
    
    with open(f"{pathName}/layer-structure.txt", "w") as file:
        for struct in structureNet:
            file.write(struct + "\n")
    
with open("Summary.txt", "w") as file:
    settings = f"Grid Search Params: \n Layers-conf: {layers_conf} \n Activation-function: {activation_functions} \n Optimizers: {optimizers} \n Lambdas: {penalities}\n Momentums: {momentums}\n Learning-rates: {learning_rates} \n\n"
    file.write(settings)
    for best in bestResults:
        file.write(best)
