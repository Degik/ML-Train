import os
import torch
import NetCup
import LoadDataCup
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import validationFunctions as vF
import IPython.display as display

from itertools import product

print("TRAINING CUP DATASET")
# HYPERPARAMETER
#interval = 0.7
learning_rate = 0.001
num_epochs = 2000
#momentum = 0.9
threshold = 0.01
#penality = 0.0005

#grid search
layers_conf = [[10, 32, 128, 128, 3]]
activation_functions = ['tanh']
optimizers = ['sgd']
penalities = [0.0005]
momentums = [0.9]
learning_rates = [0.001]
#
bestResults = []

for layers, activation, optimizerName, penality, momentum, lr in product(layers_conf, activation_functions, optimizers, penalities, momentums, learning_rates):
    # PATH
    pathTrain = "CUP/ML-CUP23-TRAIN.csv"
    pathTestInput = "CUP/ML-CUP23-TEST-INPUT.csv"
    pathTestTarget = "CUP/ML-CUP23-TEST-TARGET.csv"
    testName = f"{layers}-{optimizerName}-{activation}-{penality}-{momentum}-{lr}"
    pathName = f'modelsCup/Cup-{testName}'
    # IMPORT DATA
    dataCup = LoadDataCup.DataCup(pathTrain, pathTestInput, pathTestTarget)
    # SPLIT SET
    dataCup.splitData()
    # DATA: TENSOR, GPU, DATALOADER
    dataCup.convertToTensor()
    # MOVE TO GPU
    #device = "mps"
    #dataCup.moveToGpu(device=device)
    data_loader_train, data_loader_test = dataCup.createDataLoader()
    # CREATE NET
    structureNet = []
    # If you need to change the neurons number go to netCup.py
    print("Load regressor [net]")
    net = NetCup.NetCupRegressorVar(layers, structureNet, activation)
    #net = NetCup.NetCupCNN()
    # MOVE NET TO GPU
    #net = net.to(device)
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

    # CREATE DIR
    os.makedirs(pathName, exist_ok=True)    

    # MODEL SAVE
    """
    with open(f'{pathName}/model_parameters.txt', 'w') as file:
        file.write('Pesi layer1\n')
        file.write(str(net.layer1.weight.data) + '\n')
        file.write('Bias layer1\n')
        file.write(str(net.layer1.bias.data) + '\n')
        file.write('Pesi layer2\n')
        file.write(str(net.layer2.weight.data) + '\n')
        file.write('Bias layer2\n')
        file.write(str(net.layer2.bias.data) + '\n')
        file.write('Pesi layer3\n')
        file.write(str(net.layer3.weight.data) + '\n')
        file.write('Bias layer3\n')
        file.write(str(net.layer3.bias.data) + '\n')
        file.write('Pesi layer4\n')
        file.write(str(net.layer4.weight.data) + '\n')
        file.write('Bias layer4\n')
        file.write(str(net.layer4.bias.data) + '\n')
    """
    #Values used for graphs
    loss_values_train = []
    accuracy_values_train = []
    loss_values_test = []
    accuracy_values_test = []
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
            #
            difference = torch.abs(outputs - batch_output)
            total += batch_input.size(0)
            correct += torch.sum(difference < threshold)
        accuracy_train = correct / total
        avg_loss_train = total_loss / len(data_loader_train)
        #Add to list
        loss_values_train.append(avg_loss_train)

        total = 0
        correct = 0
        #CALCULATE ACCURACY VAL
        net.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_input, batch_output in data_loader_test:
                outputs = net(batch_input)
                loss = criterion(outputs, batch_output)
                total_loss += loss.item()
                difference = torch.abs(outputs - batch_output)
                total += batch_input.size(0)
                correct += torch.sum(difference < threshold)
            accuracy_test = correct / total
            avg_loss_test = total_loss / len(data_loader_test)
            loss_values_test.append(avg_loss_test)
        net.train()
        
        result = f'Epoch[{epoch+1}/{num_epochs}] Learning-rate: {lr}, Loss-Train: {avg_loss_train:.4f}, Loss-Test: {avg_loss_test:.4f}, Best-Accuracy-Train: {accuracy_train:.4f}, Best-Accuracy-Test: {accuracy_test:.4f}'
        print(result)
        
        #Set best accuracy
        best_accuracy_train = max(best_accuracy_train, accuracy_train)
        best_accuracy_test = max(best_accuracy_test, accuracy_test)
        #Set best loss
        best_loss_train = min(best_loss_train, avg_loss_train)
        best_loss_test = min(best_loss_test, avg_loss_test)
        #List append
        accuracy_values_train.append(accuracy_train.item())
        accuracy_values_test.append(accuracy_test.item())
        results.append(result)
        

    #Save model
    torch.save(net, f'{pathName}/model.pth')
    
    #Save best results
    bestPrint = f"{testName}\n   Best-loss-train: {best_loss_train:.4f}, Best-loss-test: {best_loss_test:.4f}, Best-Accuracy-Train: {best_accuracy_train:.4f}, Best-Accuracy-Test: {best_accuracy_test:.4f}\n"
    bestResults.append(bestPrint)
    
    # Filter loss
    #loss_values_train = utils.filterElement(loss_values_train)
    #loss_values_test = utils.filterElement(loss_values_test)
    
    # Order in the same len
    #min_len = min(len(loss_values_train), len(loss_values_test))
    #loss_values_train = loss_values_train[:min_len]
    #loss_values_test = loss_values_test[:min_len]

    #Save plot loss
    display.clear_output(wait=True)
    plt.plot(loss_values_train, label='Training Loss')
    plt.plot(loss_values_test, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.savefig(f'{pathName}/Loss.png')
    plt.clf()

    #Save plot accuracy
    display.clear_output(wait=True)
    plt.plot(accuracy_values_train, label='Accuracy Train')
    plt.plot(accuracy_values_test, label='Accuracy Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Epoch')
    plt.legend()
    plt.savefig(f'{pathName}/Accuracy-test.png')
    plt.clf()

    with open(f"{pathName}/layer-structure.txt", "w") as file:
        for struct in structureNet:
            file.write(struct + "\n")

    with open(f"{pathName}/results.txt", 'w') as file:
        for res in results:
            file.write(res + "\n")
    
with open("Summary.txt", "w") as file:
    settings = f"Grid Search Params: \n {layers_conf} \n {activation_functions} \n {optimizers} \n {penalities}\n {momentums}\n {learning_rates} \n\n"
    file.write(settings)
    for best in bestResults:
        file.write(best)