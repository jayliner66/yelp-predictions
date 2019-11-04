import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim):
        super(LSTMBaseline, self).__init__()
        self.input_dim = input_dim
        self.lstm1 = nn.LSTMCell(input_dim, 10)
        self.lstm2 = nn.LSTMCell(10, 20)
        self.linear = nn.Linear(20, 1)
    def forward(self, input, future = 0):
        
