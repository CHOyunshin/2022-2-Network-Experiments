import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import cv2
import re
import imageio.v2 as imageio
import copy

from models.CustomModelExample import MyClassifier

def main():
    # GPU Allocating
    print('==> CPU/GPU allocating..')
    print(torch.backends.mps.is_available())
    device = torch.device("mps")
    # device = torch.device("cpu")
    print( 'Device : ', device)
    # print ('Available devices : ', torch.device_count())
    # print('Selecing GPU : ',torch.get_device_name(device))
    total_t = datetime.now()

    # Load Data
    print('==> Dataset selecting..')
    # data_root = "week05_CNN/week5/labeled_data" # Set your data root
    data_root = "week05_CNN/week5/labeled_data_self" # Set your data root
    MyDataSet = CustomDataSet(data_root)

    # Data Set Split
    train_size = int(0.7 * len(MyDataSet)) # Training Size
    valid_size = int(0.2 * len(MyDataSet)) # Validation Size
    test_size = len(MyDataSet) - train_size - valid_size # Test Size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(MyDataSet, [train_size, valid_size, test_size])

    print('Number of train data : ', len(train_dataset))
    print('Number of valid data : ', len(valid_dataset))
    print('Number of test data : ', len(test_dataset))

    # Model
    print('==> Building model..')
    nb_epochs = 30 # Set an epoch 1
    torch.manual_seed(10) #Set your seed number
    BatchSize = 50 # Set a batch size

    resume = False #If you have the training networks, change this to True.
    LoadPath = './save'
    LoadEpoch = 14 #Set your best model epoch.
    if resume == True: 
        LoadName = 'best_model_'+ str(LoadEpoch) + '.pth'
        model = torch.load(os.path.join(LoadPath,LoadName))
    else:
        model = MyClassifier()
        LoadEpoch = 0
    
    model = model.to(device)
    print('Is resume : ', resume)
    print('Number of epochs : {0:^4}, Batch size: {1}'.format(nb_epochs, BatchSize))

    if device == 'cuda':
        #model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    train_dataloader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=0, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BatchSize, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0,
                                                               last_epoch=-1)

    # Start the log
    if os.path.isdir(LoadPath) == False:
        os.mkdir(LoadPath)  

    original_stdout = sys.stdout
    if resume == False:
        with open(LoadPath + '/result.txt','w') as f:
            sys.stdout = f
            print(sys._getframe().f_code.co_filename)
            print(data_root)
            print('start time : ', datetime.now())
            sys.stdout = original_stdout

    min_loss = 99999 # Allocate default value
    print('==> Training model..')
    for epoch in range(1,nb_epochs+1) :
        #Train
        start_epoch = datetime.now()
        model.train()
        train_loss = 0
        valid_total = 0
        valid_correct = 0
        #label_total = list(0. for i in range(16))
        #label_correct = list(0. for i in range(16))
        for batch_idx, (datum, targets) in enumerate(train_dataloader):
            start = datetime.now()
            datum, targets= datum.to(device), targets.long().to(device)
            img = datum.reshape([-1, 1, 64, 64]).float()
            bsz = targets.shape[0]

            # Computing loss
            out = model(img) 
            loss = criterion(out, targets)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            
            #print('Epoch: {0} | Progressing: {1} / {2} | Train loss: {3:0.6f} | Time spent: {4}, \n'.format(epoch, (batch_idx)*BatchSize + datum[0].size(0), len(train_dataset), train_loss/(batch_idx+1), datetime.now()-start))
            

        #Valid
        model.eval()
        valid_loss_value = 0
        with torch.no_grad():
            for valid_batch_idx, (datum, targets) in enumerate(valid_dataloader):
                datum, targets= datum.to(device), targets.to(device)
                img = datum.reshape([-1, 1, 64, 64]).float()
                bsz = targets.shape[0]
                #computing loss
                out = model(img) 
                _, predicted = torch.max(out,1)
                c = (predicted == targets).squeeze()
                valid_total += targets.size(0)
                valid_correct += (predicted == targets).sum().item()
                #for i in range(bsz):
                #    label = targets[i]
                #    label_correct[label] += c[i].item()
                #    label_total[label] += 1
            
                # compute loss
                valid_loss = criterion(out, targets.long())
                valid_loss_value += valid_loss.item()

        # remember minimum loss model
        is_min = valid_loss_value < min_loss
        min_loss = min(valid_loss_value, min_loss)

        if is_min == True:
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        print('Epoch: {0} | Train loss: {1:0.6f}, Valid loss: {2:0.6f}, Valid Acc: {3} / {4} | Time spent: {5}'.format(epoch, train_loss/(batch_idx+1), valid_loss_value/(valid_batch_idx+1), valid_correct, valid_total, datetime.now()-start_epoch ))
        # Save the log
        with open(LoadPath+'/result.txt','a') as f:
            sys.stdout = f
            print('Epoch: {0:^4} | Train loss: {1:0.6f}, Valid loss: {2:0.6f}, Valid Acc: {3} / {4} | Time spent: {5}'.format(epoch, train_loss/(batch_idx+1), valid_loss_value/(valid_batch_idx+1), valid_correct, valid_total, datetime.now()-start_epoch ))
            sys.stdout = original_stdout
            

    #Test
    print('==> Testing the model..')
    best_model.eval()
    test_loss_value = 0
    total = 0
    correct = 0
    label_total = list(0. for i in range(16))
    label_correct = list(0. for i in range(16))
    with torch.no_grad():
        for test_batch_idx, (datum, targets) in enumerate(test_dataloader):
            datum, targets= datum.to(device), targets.to(device)
            img = datum.reshape([-1, 1, 64, 64]).float()
            bsz = targets.shape[0]
            #computing loss
            out = best_model(img) 
            _, predicted = torch.max(out,1)
            c = (predicted == targets).squeeze()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            for i in range(bsz):
                label = targets[i]
                label_correct[label] += c[i].item()
                label_total[label] += 1
        
            # compute loss
            test_loss = criterion(out, targets.long())
            test_loss_value += test_loss.item()
    

    # Save the best model
    SaveName = 'best_model_'+ str(LoadEpoch + best_epoch) + '.pth'
    torch.save(best_model, os.path.join(LoadPath,SaveName))

    print('Best epoch : ', best_epoch)     
    for i in range(16):
        print('Accuracy of label {0} : {1:0.0f} / {2:0.0f}'.format(i, label_correct[i], label_total[i]))

    print('Test Accuracy : {0:0.3f}, {1} / {2}'.format(correct/total * 100, correct, total ) )
    print()
    print('Total Time spent with {1}: {0}'.format(datetime.now() - total_t, device))

    # Save the log
    with open(LoadPath+'/result.txt','a') as f:
        sys.stdout = f
        print('Best epoch : ', best_epoch)    
        for i in range(16):
            print('Accuracy of label {0} : {1:0.0f} / {2:0.0f}'.format(i, label_correct[i], label_total[i]))
        print('Test Accuracy : {0:0.3f}%, ({1} / {2})'.format(correct/total * 100, correct, total ) )

        sys.stdout = original_stdout

class CustomDataSet(torch.utils.data.Dataset):

    def __init__(self, data_root):
        self.data_root = data_root
        self.data = []
        self.y = []

        self.DirFolder = os.listdir(self.data_root)
    
        start=datetime.now()
        for FolderIndex in range(len(self.DirFolder)):
            self.DirData = os.listdir(self.data_root + '/' + self.DirFolder[FolderIndex])
            for DataIndex in range(len(self.DirData)):
                img= imageio.imread(self.data_root + '/'  + self.DirFolder[FolderIndex] +'/'+  self.DirData[DataIndex])
                numbers = int(re.sub(r'[^0-9]', '', self.DirFolder[FolderIndex]))
                y1 = int(numbers)
                self.data.append(img)
                self.y.append(y1)
            
        self.data = torch.from_numpy(np.array(self.data))
        self.y = torch.from_numpy(np.array(self.y))
        print('Data reading time spent : ', datetime.now()-start)


    def __getitem__(self, index):
        return self.data[index], self.y[index]
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    main()