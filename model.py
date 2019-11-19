import torch 
import numpy as np
from collections import Counter
from torch import autograd, nn
from torch.autograd import Variable
from torch import FloatTensor
from torch import optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

np.random.seed(100)

def ran(x1,x2, size):
  low1, high1, low2, high2 = x1-0.15, x1+0.15, x2-0.15, x2+0.15
  lis1 = np.random.uniform(low1, high1, size=size)
  lis2 = np.random.uniform(low2, high2, size = size)
  return list(zip(lis1,lis2))

X = xor_input = np.array(ran(1,0,700)+ ran(1,1,700) + ran(0,1,700) + ran(0,0,700))
Y = xor_output = np.array([[1]*700 + [0]*700 + [1]*700 +[0]*700]).T
X_pt = Variable(FloatTensor(X))
X_pt = X_pt.to(device = 'cuda') 
Y_pt = Variable(FloatTensor(Y), requires_grad=False)
Y_pt = Y_pt.to(device = 'cuda') 
hidden_dim = 3
#function of initializing the weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
#creating the 1 layer NN
model = nn.Sequential(nn.Linear(2, hidden_dim),
                      nn.ReLU(),
                      nn.Linear(hidden_dim, 1),
                      nn.Sigmoid())

model.apply(init_weights)
model.to(device ='cuda')
criterion = nn.BCELoss()
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# print(model, X_pt.shape, Y_pt.shape,X_pt)
num_epochs = 20
total_loss = 0.0

for _ in range(num_epochs):
    predictions = model(X_pt)
    loss_this_epoch = criterion(predictions, Y_pt)
    loss_this_epoch.backward()
    optimizer.step()
    total_loss += loss_this_epoch.item()
    ##print([float(_pred) for _pred in predictions], list(map(int, Y_pt)), loss_this_epoch.data[0])
all_results=[]
x_pred = [int(model(_x)) for _x in X_pt]
y_truth = list([int(_y[0]) for _y in Y_pt])

print(sum(np.square(np.array(y_truth)-np.array(x_pred))),total_loss/num_epochs)
