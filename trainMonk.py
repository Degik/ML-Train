import os
import torch
import NetMonk
import LoadDataMonk
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import IPython.display as display

print("TRAINING MONK DATASET")
# NET TYPE
netTypeList = {0: "Classifier", 1: "Regressor"}
netType = 1 # 0 for Classifier, 1 for Regressor
# HYPERPARAMETER
interval = 0.25
learning_rate = 0.8
hidden_units = 4
num_epochs = 500
momentum = 0.8
# PATH
pathTrain = "MONK/monks-2.train"
pathTest = "MONK/monks-2.test"
pathName = f'modelsMonk/Monk2-{hidden_units}-{learning_rate}-{netTypeList[netType]}'
# IMPORT DATA
dataMonk = LoadDataMonk.DataMonk(pathTrain, pathTest)
# DATA: TENSOR, GPU, DATALOADER
#dataMonk.convertY()
dataMonk.convertToTensor()
dataMonk.moveToGpu()
data_loader_train, data_loader_test = dataMonk.createDataLoader()
# CREATE NET
# Classifier or Regressor
if netTypeList[netType] == "Classifier":
    print("Load classifier [net]")
    net = NetMonk.NetMonkClassifier(hidden_units)
else:
    print("Load regressor [net]")
    net = NetMonk.NetMonkRegressor(interval, hidden_units)
# MOVE NET TO GPU
net = net.to("cuda:0")
# SET TYPE NET
net = net.double()
# OPTIMIZER AND CRITERION
# MSELoss for Regressor, CrossEntropyLoss for Classifier
# SGD for Regressor, Adam for Classifier
if netTypeList[netType] == "Classifier":
    print("Load CrossEntropyLoss [criterion]\nLoad Adam [optimizer]")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
else:
    print("Load MSELoss [criterion]\nLoad SGD [optimizer]")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# CREATE DIR
os.makedirs(pathName, exist_ok=True)    

# MODEL SAVE
torch.save(net, f'{pathName}/model.pth')
with open(f'{pathName}/model_parameters.txt', 'w') as file:
    file.write('Pesi layer1\n')
    file.write(str(net.layer1.weight.data) + '\n')
    file.write('Bias layer1\n')
    file.write(str(net.layer1.bias.data) + '\n')
    file.write('Pesi layer2\n')
    file.write(str(net.layer2.weight.data) + '\n')
    file.write('Bias layer2\n')
    file.write(str(net.layer2.bias.data) + '\n')

#Values used for graphs
loss_values_train = []
accuracy_values_train = []
loss_values_test = []
accuracy_values_test = []
# BEST
best_accuracy_train = 0.0
best_accuracy_test = 0.0

net.train()
for epoch in range(num_epochs):
    total_loss = 0
    total = 0
    correct = 0

    for batch_input, batch_output in data_loader_train:
        #Forward pass
        outputs = net(batch_input)
        print(outputs)
        #Training loss
        loss = None
        if netTypeList[netType] == "Classifier":
            batch_output_squeeze = batch_output.squeeze().long()
            loss = criterion(outputs, batch_output_squeeze)
        else:
            loss = criterion(outputs, batch_output)
        #Calculate total loss
        total_loss += loss.item()
        #Backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        if netTypeList[netType] == "Classifier":
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_output_squeeze).sum().item()
        else:
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_output).sum().item()
        total += batch_input.size(0)
    accuracy = correct / total
    best_accuracy_train = max(best_accuracy_train, accuracy)
    accuracy_values_train.append(accuracy)
    avg_loss = total_loss / len(data_loader_train)
    #Add to list
    loss_values_train.append(avg_loss)


    total = 0
    correct = 0
    #CALCULATE ACCURACY VAL
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_input, batch_output in data_loader_test:
            outputs = net(batch_input)
            loss = None
            if netTypeList[netType] == "Classifier":
                batch_output_squeeze = batch_output.squeeze().long()
                loss = criterion(outputs, batch_output_squeeze)
            else:
                loss = criterion(outputs, batch_output)
            total_loss += loss.item()
            if netTypeList[netType] == "Classifier":
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_output_squeeze).sum().item()
            else:
                #predicted = torch.sign(outputs)
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_output).sum().item()
            total += batch_input.size(0)
        accuracy = correct / total
        best_accuracy_test = max(best_accuracy_test, accuracy)
        accuracy_values_test.append(accuracy)
        avg_loss = total_loss / len(data_loader_test)
        loss_values_test.append(avg_loss)
    net.train()

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

result = f'Hidden_size: {hidden_units}, Learning-rate: {learning_rate}, Best-Accuracy-Train: {best_accuracy_train:.4f}, Best-Accuracy-Test: {best_accuracy_test:.4f}'
with open(f"{pathName}/result.txt", 'w') as file:
    file.write(result + "\n")
print(result)