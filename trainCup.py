import os
import torch
import NetCup
import LoadDataCup
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import IPython.display as display

print("TRAINING CUP DATASET")
# HYPERPARAMETER
interval = 0.7
learning_rate = 0.00008
num_epochs = 1000
momentum = 0.8
threshold = 0.01
# PATH
pathTrain = "CUP/ML-CUP23-TRAIN.csv"
pathTestInput = "CUP/ML-CUP23-TEST-INPUT.csv"
pathTestTarget = "CUP/ML-CUP23-TEST-TARGET.csv"
pathName = f'modelsCup/Cup-{learning_rate}-Regressor'
# IMPORT DATA
dataCup = LoadDataCup.DataCup(pathTrain, pathTestInput, pathTestTarget)
# SPLIT SET
dataCup.splitData()
# DATA: TENSOR, GPU, DATALOADER
dataCup.convertToTensor()
dataCup.moveToGpu()
data_loader_train, data_loader_test = dataCup.createDataLoader()
# CREATE NET
# If you need to change the neurons number go to netCup.py
print("Load regressor [net]")
net = NetCup.NetCupRegressor()
#net = NetCup.NetCupCNN()
# MOVE NET TO GPU
net = net.to("cuda:0")
# SET TYPE NET
net = net.double()
# OPTIMIZER AND CRITERION
# MSELoss for Regressor
# SGD for Regressor
print("Load MSELoss [criterion]\nLoad SGD [optimizer]")
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

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
    
    result = f'Epoch[{epoch+1}/{num_epochs}] Learning-rate: {learning_rate}, Loss-Train: {avg_loss_train:.4f}, Loss-Test: {avg_loss_test:.4f}, Best-Accuracy-Train: {accuracy_train:.4f}, Best-Accuracy-Test: {accuracy_test:.4f}'
    print(result)
    
    #Set best accuracy
    best_accuracy_train = max(best_accuracy_train, accuracy_train)
    best_accuracy_test = max(best_accuracy_test, accuracy_test)
    
    #List append
    accuracy_values_train.append(accuracy_train.item())
    accuracy_values_test.append(accuracy_test.item())
    results.append(result)
    

#Save model
torch.save(net, f'{pathName}/model.pth')

#Save plot loss
display.clear_output(wait=True)
plt.plot(loss_values_train, label='Training Loss')
plt.plot(loss_values_test, label = 'Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
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


with open(f"{pathName}/results.txt", 'w') as file:
    for res in results:
        file.write(res + "\n")
    bestPrint = f"Best-Accuracy-Train: {best_accuracy_train:.4f}, Best-Accuracy-Test: {best_accuracy_test:.4f}"
    file.write(bestPrint + "\n")