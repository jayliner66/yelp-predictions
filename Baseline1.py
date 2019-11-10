
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import json

np.random.seed(42)

def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

x_train = []# numpy array of triples of inputs
y_correct = []# numpy array of pairs of outputs

business_dict = {}
review_dict = {}

for business in json_readline("yelp_dataset/business.json"):
    business["reviews"] = []
    business_dict[business["business_id"]] = business
#for review in json_readline("yelp_dataset/review.json"):
#    review_dict[review["review_id"]] = review
#    if review["business_id"] in business_dict:
#        business_dict[review["business_id"]]["reviews"].append(review["review_id"])
print("Done with part 1")




# first review
for id in business_dict:
    business = business_dict[id]
    x_train.append(business["review_count"])
    x_train.append(business["is_open"])
    y_correct.append(business["stars"])
print("Done with part 2")
x_train = np.asarray(x_train, dtype=np.float32).reshape(-1, 2)
print(x_train)
print(x_train.shape)

y_correct = np.asarray(y_correct, dtype=np.float32).reshape(-1, 1)
print(y_correct.shape)
class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out

input_dim = 2
output_dim = 1


model = LinearRegressionModel(input_dim,output_dim)# create our model just as we do in Scikit-Learn / C / C++//

criterion = nn.MSELoss()# Mean Squared Loss
l_rate = 0.000001
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent

epochs = 2000

for epoch in range(epochs):

    epoch +=1
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_correct))

    #clear grads
    optimiser.zero_grad()
    #forward to get predicted values
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()# back props
    optimiser.step()# update the parameters
    print('epoch {}, loss {}'.format(epoch,loss.data))

predicted = model.forward(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(x_train, y_correct, 'go', label = 'from data', alpha = .5)
plt.plot(x_train, predicted, label = 'prediction', alpha = 0.5)
plt.legend()
plt.show()
print(model.state_dict())
