# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:45:16 2026

@author: mihal
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


num_layers = 2       #hiperparametri
hidden_size = 256
num_classes = 5
learning_rate = 0.001
batch_size = 64
num_epochs = 10


def load_training_sample(filepath):
    data = np.load(filepath)

    # inputs — one rgb image per sensor, shape (F, T, 3)
    accel = data['accel_rgb']   # float or uint8, cast as needed
    gyro  = data['gyro_rgb']
    mag   = data['mag_rgb']

    # labels — shape (T, 5)
    y = data['labels']

    # time axis if needed for alignment checks
    times = data['accel_times']

    return accel, gyro, mag, y, times


class BiLSTM(nn.Module):
    def __init__(self, sample_files):
        super(BiLSTM, self).__init__()
        
        self.sample_files = sample_files
        self.n_samples = len(sample_files)
        self.samples = np.zeros((self.n_samples,))
        self.labels = np.zeros((self.n_samples,))


        self.lstm = nn.LSTM(self.n_samples, hidden_size, num_layers, bidirectional=True)    #batch_first=True
        self.fc = nn.Linear(hidden_size*2, num_classes)  #fully connected layer
        
        
    def forward(self, x):
        h0 = torch.zeros(num_layers*2, x.size(0), hidden_size)  #*2 ker bidirectional
        c0 = torch.zeros(num_layers*2, x.size(0), hidden_size)  #*2 ker bidirectional

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])    #zadnji hidden state gre v out 
        
        return out
    

    def get_training_samples(self, sample_files):
        for i in range(self.n_samples):
            sample_path = sample_files[i]
            accel, gyro, mag, y, times = load_training_sample(sample_path)
            
            F = accel.shape[0] 
            T = accel.shape[1]  
            
            accel_flat = accel.reshape(F*3, T).T 
            gyro_flat  = gyro.reshape(F*3, T).T
            mag_flat   = mag.reshape(F*3, T).T 
            
            x = np.concatenate([accel_flat, gyro_flat, mag_flat], axis=-1)  #en input
            
            self.samples[i] = x
            self.labels[i] = y
            
            
def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)        
    
    for epoch in range(num_epochs):
        data = ...
        targets = ...
        
        scores = model(data)    #forward
        loss = criterion(scores, targets)
        
        optimizer.zero_grad()   #backward
        loss.backward()
        
        optimizer.step()
        
        
    
def main():
    sample_files = Path('data').glob('*.npz')

    bilstm = BiLSTM(sample_files)
    bilstm.get_training_samples()
    
    train_model(bilstm)
    

if __name__ == "__main__":
    main()