import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


#TODO: add X_train, y_train
#X_train =
#y_train =

input_dim = 2
hidden_dim = 30
layer_num = 2
output_dim = 1
seq_len = 3


learning_rate = 0.01
num_epochs = 2000


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(LSTMBaseline, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, input):
        self.hidden = (torch.zeros(layer_num, input.size(0), hidden_dim), torch.zeros(layer_num, input.size(0), hidden_dim))
        
        output, self.hidden = self.lstm(input.view(input.size(0), seq_len, -1))
        y_pred = self.linear(output[:,-1].view(input.size(0), -1))
        
        return y_pred.view(-1)     
        
model = LSTMBaseline(input_dim, seq_len)

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    
    y_pred = model(X_train)
    
    loss = loss_fn(y_pred, y_train)
    
    print("Epoch: ", t, "MSE: ", loss.item())
    
    hist[t] = loss.item()
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    

plt.plot(y_pred.detach().numpy(), label = "Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

plt.hist(y_pred.detach().numpy(), density=True, bins=30)

plt.show()