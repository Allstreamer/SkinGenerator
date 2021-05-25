import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 1000
batchsize = 32

transform = transforms.ToTensor()

class Clasifiyer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 10*10),
            nn.LeakyReLU(),
            nn.Linear(10*10, 5*5),
            nn.LeakyReLU(),
            nn.Linear(5*5, 10)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
network = Clasifiyer().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())

for epoch in range(epochs):
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = network(inputs)
        
        loss = criterion(outputs,labels)
        loss.backward()
        
        optimizer.step()
    print(f"Epoch: {epoch} Loss:{loss}")