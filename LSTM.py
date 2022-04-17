# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:17:25 2022

@author: Riley
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
X = np.load('data.npy', allow_pickle=True)
numSamples = X.shape[0]
y2, y3 = [0,1,1]*(numSamples//3), [0,1,2]*(numSamples//3)
y2, y3 = np.array(y2).astype('float'), np.array(y3).astype('float')

features_train, features_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Dataset
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, y=None):
        self.features = features
        self.y = y
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).reshape(475,13), torch.Tensor(np.array(self.y[idx]))  # (1,13,475)

# Create an LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(FeatureDataset(features_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(FeatureDataset(features_test, y_test), batch_size=batch_size, shuffle=False)

model = LSTM(input_size=13, hidden_size=100, num_layers=1, num_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
    
num_epochs = 500
for epoch in range(num_epochs):
    correct_train = correct_test = 0
    for i, batch in enumerate(train_loader):
        model.train()
        x = batch[0].float().to(device)
        y = batch[1].float().to(device)
        y_pred = model(x).squeeze()
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct_train += (y_pred.round() == y).sum().item()
    for i, batch in enumerate(test_loader):
        model.eval()
        x = batch[0].float().to(device)
        y = batch[1].float().to(device)
        y_pred = model(x).squeeze()
        correct_test += (y_pred.round() == y).sum().item()
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}:')
        print(f'Loss: {loss.item():.2f}, Train Accuracy: {correct_train/len(train_loader.dataset):.2f}, Test Accuracy: {correct_test/len(test_loader.dataset):.2f}\n')