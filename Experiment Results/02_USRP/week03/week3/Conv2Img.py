import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import cv2

def main():
    data_root = "./week03\week3\IQ_signal"  # Set your data root
    write_root = "./week03\week3\spectrogram" # Set your save file root    
    # data_root = "./IQ_signal"  # Set your data root
    # write_root = "./spectrogram" # Set your save file root

    if os.path.isdir(write_root) == False:
        os.mkdir(write_root)

    MyDataSet = CustomDataSet(data_root)
    data_length = MyDataSet.__len__()
    for i in range(data_length):
        [datum, target]= MyDataSet.__getitem__(i) 
        name = target[0]
        cv2.imwrite(write_root +'/' + name[0:-4] +'.png', datum[0])


class CustomDataSet(torch.utils.data.Dataset):

    def __init__(self, data_root):
        self.data_root = data_root
        self.data = []
        self.y = []

        self.DirFolder = os.listdir(self.data_root)

        totalnumberOfData = 0
        start=datetime.now()
        for FolderIndex in range(len(self.DirFolder)):
            numberofData = 0
            self.DirData = os.listdir(self.data_root + '/' + self.DirFolder[FolderIndex])
            totalnumberOfData += len(self.DirData)
            numberOfData = len(self.DirData)
            print('Try Folder : ', FolderIndex)
            #label = pd.read_csv(label_root + '/' + self.DirFolder[FolderIndex] +'.csv')
            for DataIndex in range(len(self.DirData)):
                df_= pd.read_csv(self.data_root + '/'  + self.DirFolder[FolderIndex] +'/'+  self.DirData[DataIndex],header=None)
                #print(np.shape(df_))
                #iq_signal= df_.iloc[:,0]+1j*df_.iloc[:,1]
                iq_signal= df_.iloc[0,:]+1j*df_.iloc[1,:]
                (S,f,t)= plt.mlab.specgram(iq_signal, Fs=10e7, NFFT=64, noverlap=-14)
                S=abs(S)
                S=10*np.log10(S)
                S= (((S-np.min(S)) / (np.max(S)-np.min(S))) * 255).astype(np.uint8)
                S= S[::-1]
                #y1 = label.iloc[DataIndex,1]
                y1 = self.DirData[DataIndex]
                self.data.append([S])
                self.y.append([y1])
            print('Folder Name in given IQ Signal file',self.DirFolder[FolderIndex],'there are # of files data :', numberOfData)
            
        #self.data = torch.from_numpy(np.array(self.data))
        #self.y = torch.from_numpy(np.array(self.y))
        print('Total Data : ', totalnumberOfData)
        print('Data reading time spent : ', datetime.now()-start)


    def __getitem__(self, index):
        return self.data[index], self.y[index]
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    main()