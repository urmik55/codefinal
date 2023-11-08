import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
from sklearn.model_selection import train_test_split
from pyDOE import lhs  # Latin Hypercube Sampling

# Define a function to split the data into training, validation, and test sets
def split_data(x, y, train_size=0.8, val_size=0.1, random_state=42):
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(1 - train_size), random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(val_size / (val_size + (1 - train_size))), random_state=random_state)
    return x_train, y_train, x_val, y_val, x_test, y_test

# Define the neural network class
class ResNet(nn.Module):
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.layers = layers  # Store the layers as an instance variable
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        a = x.float()
        for i in range(len(self.layers) - 2):  # Use self.layers
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss_BC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC

    def loss_PDE(self, x_PDE, device):
        g = x_PDE.clone()
        g.requires_grad = True
        f = self.forward(g)
        f_x = autograd.grad(f, g, torch.ones([x_PDE.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        f_xx = autograd.grad(f_x, g, torch.ones([x_PDE.shape[0], 1]).to(device), create_graph=True)[0]
        return self.loss_function(f_xx, self.PDE(g))  # Call PDE function within the class

    def loss(self, x_BC, y_BC, x_PDE, device):
        loss_bc = self.loss_BC(x_BC, y_BC)
        loss_pde = self.loss_PDE(x_PDE, device)
        return loss_bc + loss_pde

    # Define PDE function within the class
    def PDE(self, x):
        return -1 * (np.pi ** 2) * torch.sin(np.pi * x)

# Function to evaluate metrics
def evaluate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ssr = np.sum((y_pred - y_true) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / sst)
    return mse, rmse, mae, r2

def Poison():
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name())

    steps = 5000
    lr = 1e-3
    layers = [1, 50, 50, 20, 50, 50, 1]
    min_val = -1
    max_val = 1
    total_points = 500
    Nu = 2
    Nf = 250

    def f_BC(x):
        return 1 - torch.abs(x)

    def f_real(x):
        return torch.sin(np.pi * x)

    def PDE(x):
        return -1 * (np.pi ** 2) * torch.sin(np.pi * x)

    x = torch.linspace(min_val, max_val, total_points).view(-1, 1)
    y = f_real(x)
    print(x.shape, y.shape)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)

    y_BC = f_BC(x_train).to(device)
    x_PDE = x_train.clone().to(device)

    model = ResNet(layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

    for i in range(steps):
        yh = model(x_PDE)
        loss = model.loss(x_train, y_BC, x_PDE, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % (steps / 10) == 0:
            print(loss)

    yh = model(x.to(device))
    y = f_real(x)
    mse, rmse, mae, r2 = evaluate_metrics(y.cpu().detach().numpy(), yh.cpu().detach().numpy())
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}')

    g = x.to(device).clone()
    g.requires_grad = True
    f = model(g)
    f_x = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
    f_x_plot = f_x.cpu().detach().numpy()

    y_plot = y.cpu().detach().numpy()
    yh_plot = yh.cpu().detach().numpy()

    fig, ax1 = plt.subplots()
    ax1.plot(x, y_plot, color='blue', label='Real u(x)')
    ax1.plot(x, yh_plot, color='red', label='Predicted u(x)')
    ax1.plot(x, f_x_plot, color='green', label="Computed u'(x)")
    ax1.set_xlabel('x', color='black')
    ax1.set_ylabel('f(x)', color='black')
    ax1.tick_params(axis='y', color='black')
    ax1.legend(loc='upper left')
    plt.savefig("Results/Poison_pred.png")
    plt.show()

if __name__ == "__main__":
    Poison()
