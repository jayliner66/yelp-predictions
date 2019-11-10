import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


#TODO: add X_train, y_train, X_test, y_test
#X_train = 
#y_train = 
#X_test = 
#y_test = 

input_dim = 1
hidden_dim = 30
layer_num = 2
output_dim = 1
seq_len = len(X_train[0])


learning_rate = 0.001
num_epochs = 500
batch_size = 20


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
    sum_loss=0
    n_batches = 0
    X_batches = torch.split(X_train, batch_size)
    y_batches = torch.split(y_train, batch_size)
    batches = zip(X_batches, y_batches)
    for (X_batch, y_batch) in batches:
        model.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = loss_fn(y_pred, y_batch)
        
        sum_loss+=loss.item()
        n_batches+=1
        
        optimizer.zero_grad()
    
        loss.backward()
    
        optimizer.step()
    sum_loss/=(n_batches)
    
    print("Epoch: ", t, "MSE: ", sum_loss)
    
    hist[t] = sum_loss
    

y_pred = model(X_train)

plt.plot(y_pred.detach().numpy(), label = "Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

y_test_pred=model(X_test)

plt.plot(y_test_pred.detach().numpy(), label="Test Preds")
plt.plot(y_test.detach().numpy(), label="Data")
plt.legend()
plt.show()