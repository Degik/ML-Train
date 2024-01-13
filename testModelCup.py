import os
import time
import utils
import torch
import NetCup
import statistics
import numpy as np
import LoadDataCup
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import IPython.display as display

from itertools import product



print("TRAINING CUP DATASET")
# PATH
pathTrain = "CUP/ML-CUP23-TRAIN.csv"
pathTestInput = "CUP/ML-CUP23-TEST-INPUT.csv"
pathTestTarget = "CUP/ML-CUP23-TEST-TARGET.csv"
seed = int(time.time()%150)
# HYPERPARAMETER
num_epochs = 5000
#momentum = 0.9
threshold = 0.01
#penality = 0.0005

#Netowrks
network_configs = [
    {"layers": [10, 256, 256, 300, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0002, "momentum": 0.8, "learning_rate": 0.003},
    {"layers": [10, 256, 256, 300, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0005, "momentum": 0.8, "learning_rate": 0.003},
    {"layers": [10, 300, 256, 256, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0001, "momentum": 0.8, "learning_rate": 0.003},
    {"layers": [10, 512, 512, 600, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0001, "momentum": 0.8, "learning_rate": 0.003},
    {"layers": [10, 256, 256, 300, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0002, "momentum": 0.9, "learning_rate": 0.001},
    {"layers": [10, 512, 512, 600, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0005, "momentum": 0.8, "learning_rate": 0.003},
    # Add here others config
]
#
numberTest = len(network_configs)
bestResults = []

# IMPORT DATA
dataCup = LoadDataCup.DataCup(pathTrain)
#Split Data
dataCup.splitData()
# DATA: TENSOR, GPU, DATALOADER
dataCup.convertToTensor()
# MOVE TO GPU
device = "cuda:0"
dataCup.moveToGpu(device=device)
###
data_loader_train, data_loader_test = dataCup.createDataLoader()

#We will use this list for save all data for all training epoch and then calculate the mean
history_train_loss = []
history_train_mee = []
history_test_loss = []
history_test_mee = []


for number, config in enumerate(network_configs):
    #Net settings
    layers = config['layers']
    activation = config['activation']
    optimizerName = config['optimizer']
    penality = config['penality']
    momentum = config['momentum']
    lr = config['learning_rate']
    # PATH
    testName = f"{layers}-{optimizerName}-{activation}-{penality}-{momentum}-{lr}"
    pathName = f'modelsCup/Cup-{testName}'
    bestResults.append(testName + "\n")

    # CREATE DIR
    os.makedirs(pathName, exist_ok=True)
    
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
    
    #Values used for graphs
    loss_values_train = []
    accuracy_testues_train = []
    loss_values_test = []
    accuracy_testues_test = []
    # Distance list
    euclidean_distance_train = []
    euclidean_distance_test = []
    # Distances list
    euclidean_distances_train = []
    euclidean_distances_test = []
    # BEST
    best_accuracy_train = 0.0
    best_accuracy_test = 0.0
    best_loss_train = 100.0
    best_loss_test = 100.0
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
            #Take distance
            distance = utils.euclidean_distance_loss(batch_output, outputs)
            #Add distance to others
            euclidean_distance_train.append(distance.item())
        avg_loss_train = total_loss / len(data_loader_train)
        loss_values_train.append(avg_loss_train)

        total = 0
        correct = 0
        #CALCULATE ACCURACY test
        net.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_input, batch_output in data_loader_test:
                outputs = net(batch_input)
                loss = criterion(outputs, batch_output)
                total_loss += loss.item()
                # Take distance
                # Return the mean for the distance inside the batch
                distance = utils.euclidean_distance_loss(batch_output, outputs)
                # Add distance to others
                # Take each distance for each bacth_size
                euclidean_distance_test.append(distance.item())
                
            # Mean distance
            # Calculate the mean between all batch inside the epoch
            mean_distance_train = statistics.mean(euclidean_distance_train)
            mean_distance_test = statistics.mean(euclidean_distance_test)
            # Save the mean for the current epoch in the list
            euclidean_distances_train.append(mean_distance_train)
            euclidean_distances_test.append(mean_distance_test)
            # Mean loss
            avg_loss_train = total_loss / len(data_loader_train)
            avg_loss_test = total_loss / len(data_loader_test)
            loss_values_test.append(avg_loss_test)
        net.train()
        
        result = f'Epoch[{epoch+1}/{num_epochs}] Learning-rate: {lr}, Loss-Train: {avg_loss_train:.4f}, Loss-test: {avg_loss_test:.4f} MEE-Train: {mean_distance_train:.4f}, MEE-test: {mean_distance_test:.4f}'
        print(f"Test[{number+1}/{numberTest}] --> " + result)
        
        #Set best loss
        best_loss_train = min(best_loss_train, avg_loss_train)
        best_loss_test = min(best_loss_test, avg_loss_test)
        #List append
        results.append(result) 

    #END EPOCHS
    
    #History loss
    history_train_loss.append(loss_values_train)
    history_test_loss.append(loss_values_test)
    #History distance
    history_train_mee.append(euclidean_distances_train)
    history_test_mee.append(euclidean_distances_test)
    
    #Save best results
    bestPrint = f'     Best-loss-train: {best_loss_train:.4f}, Best-loss-test: {best_loss_test:.4f} \n'
    bestResults.append(bestPrint)
    
    with open(f"{pathName}/results.txt", 'w') as file:
        for res in results:
            file.write(res + "\n")
            
    #Last loss
    last_train_loss = loss_values_train[-1]
    last_test_loss = loss_values_test[-1]
    #Last MEE
    last_mee_train = euclidean_distance_train[-1]
    last_mee_test = euclidean_distance_test[-1]
    #Adding best results
    bestPrint = f"     Mean-Last-Epoch-Train: {last_train_loss:.4f}, Mean-Last-Epoch-test: {last_test_loss:.4f}, MEE-Train: {last_mee_train:.4f}, MEE-test: {last_mee_test:.4f}\n"
    bestResults.append(bestPrint)
    
    #Save plot loss
    display.clear_output(wait=True)
    plt.plot(loss_values_train, label='Training Loss')
    plt.plot(loss_values_test, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Mean-Loss per Epoch')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.savefig(f'{pathName}/Mean-Loss.png')
    plt.clf()
    
    #Save plot loss
    display.clear_output(wait=True)
    plt.plot(euclidean_distances_train, label='MEE-Training')
    plt.plot(euclidean_distances_test, label = 'MEE-Test')
    plt.xlabel('Epoch')
    plt.ylabel('MEE')
    plt.title(f'MEE per Epoch')
    plt.ylim([0, 3.0])
    plt.legend()
    plt.savefig(f'{pathName}/MEE.png')
    plt.clf()
    
    #Save model
    torch.save(net, f'{pathName}/model.pth')
    
    with open(f"{pathName}/layer-structure.txt", "w") as file:
        for struct in structureNet:
            file.write(struct + "\n")
#END MODELS CONFIG

#MEAN ALL DATA
#Mean history
mean_train_loss = np.mean(history_train_loss, axis=0)
mean_test_loss = np.mean(history_test_loss, axis=0)
mean_train_mee = np.mean(history_train_mee, axis=0)
mean_test_mee = np.mean(history_test_mee, axis=0)
#Last loss
last_mean_train_loss = mean_train_loss[-1]
last_mean_test_loss = mean_test_loss[-1]
#Last MEE
last_mean_train_mee = mean_train_mee[-1]
last_mean_test_mee = mean_test_mee[-1]

bestPrint = f"MERGE-MODELS --> Mean-Last-Epoch-Train: {last_mean_train_loss:.4f}, Mean-Last-Epoch-test: {last_mean_test_loss:.4f}, MEE-Train: {last_mean_train_mee:.4f}, MEE-test: {last_mean_test_mee:.4f}\n"
bestResults.append(bestPrint)

#Save plot loss
display.clear_output(wait=True)
plt.plot(mean_train_loss, label='Training Loss')
plt.plot(mean_test_loss, label = 'Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Mean-Loss per Epoch')
plt.ylim([0, 1.2])
plt.legend()
plt.savefig(f'modelsCup/Mean-Loss-Merge.png')
plt.clf()

#Save plot loss
display.clear_output(wait=True)
plt.plot(mean_train_mee, label='MEE-Training')
plt.plot(mean_test_mee, label = 'MEE-Test')
plt.xlabel('Epoch')
plt.ylabel('MEE')
plt.title(f'MEE per Epoch')
plt.ylim([0, 3.0])
plt.legend()
plt.savefig(f'modelsCup/MEE-Merge.png')
plt.clf()


with open("modelsCup/Summary.txt", "w") as file:
    file.write("Netoworks config: " + "\n")
    for config in network_configs:
        file.write(f"  {str(config)} \n")
    file.write("\n\n\n")
    for best in bestResults:
        file.write(best)
