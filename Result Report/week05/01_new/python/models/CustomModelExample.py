import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.layer2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)

        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 16)

    def forward(self, x):
        #print(x.size()) # Check data size
        x = F.max_pool2d(F.relu(self.layer1(x)), (2,2))
        #print(x.size())
        x = F.max_pool2d(F.relu(self.layer2(x)), (2,2))
        #print(x.size())
        x = torch.flatten(x,1)
        #print(x.size())
        x = F.relu(self.linear1(x))
        #print(x.size())
        x = self.linear2(x)
        #print(x.size())
        return x
#net = MyClassifier()
#print(net)

#td = torch.rand(50,1,64,64)
#out = net(td)
#print(out.size())

