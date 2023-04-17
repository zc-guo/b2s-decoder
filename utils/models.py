#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:57:23 2021

@author: adamguo
"""

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define models
class LSTM_basic(nn.Module):
    def __init__(self, n_channels = 62, hidden_size = 256, n_layers = 3, dropout = 0.2):
        super(LSTM_basic, self).__init__()
        
        self.input_size = n_channels
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Default: use 3 stacked LSTM layers with input_size = 62 (EEG channels) and hidden_size = 256
        self.rnn1 = nn.LSTM(input_size = n_channels, hidden_size = hidden_size,
                            num_layers = n_layers, batch_first = True, dropout = dropout)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        h_0 = torch.randn(self.n_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.n_layers, x.size(0), self.hidden_size).to(device)
        
        x, (h_n, c_n) = self.rnn1(x, (h_0, c_0))
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
