import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


LSTM = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)
optimizer = optim.SGD(LSTM.parameters(), lr=0.01, momentum=0.9)
criterion = nn.L1Loss()

# batch_size=8, sequence_length=32, dim=16
input_data = torch.randn(32, 16)

for i in range(1000):

    input = [input_data[0], input_data[2], input_data[4], input_data[8]]
    output, (hn, cn) = LSTM(input_data)


    loss = criterion(output, input_data)
    print(loss.sum())
    loss.backward()
    optimizer.step()

print('done')
