import torch
import torch.autograd as autograd
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling

from Subfunctions import evaluate_metrics
from Subfunctions.dynamic_system_simulation import *

def Pinn1():
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)

    # PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)

    # Device configuration
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(device)

    # Define the f_real function
    def f_real(x, t):
        return torch.exp(-t) * (torch.sin(np.pi * x))

    # %%
    steps = 20000
    lr = 1e-3
    layers = np.array([2, 32, 32, 1])  # hidden layers
    # To generate new data:
    x_min = -1
    x_max = 1
    t_min = 0
    t_max = 1
    total_points_x = 200
    total_points_t = 100
    # Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
    Nu = 100
    Nf = 10000

    def plot3D(x, t, y):
        x_plot = x.squeeze(1)
        t_plot = t.squeeze(1)
        X, T = torch.meshgrid(x_plot, t_plot)
        F_xt = y
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
        fig.colorbar(cp)  # Add a colorbar to a plot
        ax.set_title('F(x,t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.show()
        ax = plt.axes(projection='3d')
        ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('f(x,t)')
        plt.show()

    # Generate data
    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    # Create the mesh
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    # Evaluate real function
    y_real = f_real(X, T)
    plot3D(x, t, y_real)  # f_real was defined previously (function)

    # Shuffle your data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_train_Nu = X[indices[:Nu], :]
    Y_train_Nu = y_real[indices[:Nu], :]
    X_train_Nf = X[indices[Nu:Nu + Nf], :]
    Y_train_Nf = y_real[indices[Nu:Nu + Nf], :]
    X_test = X[indices[Nu + Nf:], :]
    Y_test = y_real[indices[Nu + Nf:], :]

    def plot3D_Matrix(x, t, y):
        X, T = x, t
        F_xt = y
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
        fig.colorbar(cp)  # Add a colorbar to a plot
        ax.set_title('F(x,t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.savefig(f"Results\\Diffusion_pred1.png")
        plt.show()
        ax = plt.axes(projection='3d')
        ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('f(x,t)')
        plt.savefig(f"Results\\Diffusion_pred2.png")
        plt.show()

    def f_real(x, t):
        return torch.exp(-t) * (torch.sin(np.pi * x))

    class Resnet(nn.Module):
        # Neural Network
        def __init__(self, layers):
            super().__init__()  # call __init__ from parent class
            'activation function'
            self.activation = nn.Tanh()
            'loss function'
            self.loss_function = nn.MSELoss(reduction='mean')
            'Initialise neural network as a list using nn.Modulelist'
            self.linears = nn.ModuleList(
                [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.iter = 0  # For the Optimizer
            'Xavier Normal Initialization'
            for i in range(len(layers) - 1):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
                # set biases to zero
                nn.init.zeros_(self.linears[i].bias.data)

        'foward pass'

        def forward(self, x):
            if torch.is_tensor(x) != True:
                x = torch.from_numpy(x)
            a = x.float()
            for i in range(len(layers) - 2):
                z = self.linears[i](a)
                a = self.activation(z)
            a = self.linears[-1](a)
            return a

        'Loss Functions'

        # Loss BC

        def lossBC(self, x_BC, y_BC):
            loss_BC = self.loss_function(self.forward(x_BC), y_BC)
            return loss_BC

        # Loss PDE

        def lossPDE(self, x_PDE):
            g = x_PDE.clone()
            g.requires_grad = True  # Enable differentiation
            f = self.forward(g)
            f_x_t = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(
                device), retain_graph=True, create_graph=True)[0]  # first derivative
            f_xx_tt = autograd.grad(f_x_t, g, torch.ones(g.shape).to(
                device), create_graph=True)[0]  # second derivative
            # we select the 2nd element for t (the first one is x) (Remember the input X=[x,t])
            f_t = f_x_t[:, [1]]
            # we select the 1st element for x (the second one is t) (Remember the input X=[x,t])
            f_xx = f_xx_tt[:, [0]]
            f = f_t - f_xx + torch.exp(-g[:, 1:]) * (torch.sin(np.pi *
                                                               g[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * g[:, 0:1]))
            return self.loss_function(f, f_hat)

        def loss(self, x_BC, y_BC, x_PDE):
            loss_bc = self.lossBC(x_BC, y_BC)
            loss_pde = self.lossPDE(x_PDE)
            return loss_bc + loss_pde

        # Optimizer              X_train_Nu,Y_train_Nu,X_train_Nf
        def closure(self):
            optimizer.zero_grad()
            loss = self.loss(X_train_Nu, Y_train_Nu, X_train_Nf)
            loss.backward()
            self.iter += 1
            if self.iter % 100 == 0:
                loss2 = self.lossBC(X_test, Y_test)
                print("Training Error:", loss.detach().cpu().numpy(),
                      "---Testing Error:", loss2.detach().cpu().numpy())
            return loss

    # %% [markdown]
    # # Generate data

    # %%
    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    # Create the mesh
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    # Evaluate real function
    y_real = f_real(X, T)
    plot3D(x, t, y_real)  # f_real was defined previously(function)

    # %%
    print(x.shape, t.shape, y_real.shape)
    print(X.shape, T.shape)

    # %% [markdown]
    # ## Prepare Data

    # %%
    # Transform the mesh into a 2-column vector
    x_test = torch.hstack(
        (X.transpose(1, 0).flatten()[:, None], T.transpose(1, 0).flatten()[:, None]))
    # Colum major Flatten (so we transpose it)
    y_test = y_real.transpose(1, 0).flatten()[:, None]
    # Domain bounds
    lb = x_test[0]  # first value
    ub = x_test[-1]  # last value
    print(x_test.shape, y_test.shape)
    print(lb, ub)

    # ## Training Data

    # Initial Condition
    # Left Edge: x(x,0)=sin(x)->xmin=<x=<xmax; t=0
    # First column # The [:,None] is to give it the right dimension
    left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    left_Y = torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)

    # First row # The [:,None] is to give it the right dimension
    bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))
    bottom_Y = torch.zeros(bottom_X.shape[0], 1)

    # Last row # The [:,None] is to give it the right dimension
    top_X = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))
    top_Y = torch.zeros(top_X.shape[0], 1)
    # Get all the training data into the same dataset
    X_train = torch.vstack([left_X, bottom_X, top_X])
    Y_train = torch.vstack([left_Y, bottom_Y, top_Y])
    # Choose(Nu) points of our available training data:
    idx = np.random.choice(X_train.shape[0], Nu, replace=False)
    X_train_Nu = X_train[idx, :]
    Y_train_Nu = Y_train[idx, :]
    # Collocation Points (Evaluate our PDe)
    # Choose(Nf) points(Latin hypercube)
    X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and t
    # Add the training poinst to the collocation points
    X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))

    print("Original shapes for X and Y:", X.shape, y_real.shape)
    print("Boundary shapes for the edges:",
          left_X.shape, bottom_X.shape, top_X.shape)
    print("Available training data:", X_train.shape, Y_train.shape)
    print("Final training data:", X_train_Nu.shape, Y_train_Nu.shape)
    print("Total collocation points:", X_train_Nf.shape)

    torch.manual_seed(123)
    # Store tensors to GPU
    X_train_Nu = X_train_Nu.float().to(device)  # Training Points (BC)
    Y_train_Nu = Y_train_Nu.float().to(device)  # Training Points (BC)
    X_train_Nf = X_train_Nf.float().to(device)  # Collocation Points
    f_hat = torch.zeros(X_train_Nf.shape[0], 1).to(device)  # to minimize function

    X_test = x_test.float().to(device)  # the input dataset (complete)
    Y_test = y_test.float().to(device)  # the real solution

    # Create Model
    PINN = Resnet(layers)
    PINN.to(device)
    print(PINN)
    params = list(PINN.parameters())
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr, amsgrad=False)
    '''
    'L-BFGS Optimizer'
    optimizer = torch.optim.LBFGS(PINN.parameters(), lr=lr,
                                  max_iter = steps,
                                  max_eval = None,
                                  tolerance_grad = 1e-05,
                                  tolerance_change = 1e-09,
                                  history_size = 100,
                                  line_search_fn = 'strong_wolfe')'''
    start_time = time.time()

    for i in range(steps):
        if i == 0:
            print("Training Loss-----Test Loss")
        # use mean squared error
        loss = PINN.loss(X_train_Nu, Y_train_Nu, X_train_Nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % (steps / 10) == 0:
            with torch.no_grad():
                test_loss = PINN.lossBC(X_test, Y_test)
            print(loss.detach().cpu().numpy(), '---',
                  test_loss.detach().cpu().numpy())

    # ### Plots

    y1 = PINN(X_test)

    x1 = X_test[:, 0]
    t1 = X_test[:, 1]
    [mse, rmse, mae, r2] = evaluate_metrics(x1, t1)
    np.save('NPY\\m4.npy', np.array([mse, rmse, mae, r2]))

    arr_x1 = x1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
    arr_T1 = t1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
    arr_y1 = y1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
    arr_y_test = y_test.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()

    plot3D_Matrix(arr_x1, arr_T1, arr_y1)

    plot3D_Matrix(X, T, y_real)


if __name__ == "__main__":
    Pinn1()
