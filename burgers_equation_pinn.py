import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pyDOE import lhs
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd

from Subfunctions import evaluate_metrics

def Pinn2():
    """
    Train a Physics-Informed Neural Network (PINN) model.

    This function implements a PINN model and performs training, validation, and testing with proper data splitting.
    """

    # Set default data type to float32
    torch.set_default_dtype(torch.float)

    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name())

    # Define hyperparameters
    num_steps = 20000
    learning_rate = 1e-3
    layers = np.array([2, 32, 32, 1])  # Hidden layers

    # Define data generation parameters
    x_min = -1
    x_max = 1
    t_min = 0
    t_max = 1
    total_points_x = 200
    total_points_t = 100

    # Number of training points and collocation points
    Nu = 100  # Define the number of training points
    num_collocation_points = 10000

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    def plot3D(x, t, y):
        """
        Plot a 3D surface graph.

        Args:
            x (torch.Tensor): X-coordinate data.
            t (torch.Tensor): T-coordinate data.
            y (torch.Tensor): Function values.
        """
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
        plt.savefig(f"Results\\Burgers1.png")
        plt.show()

        ax = plt.axes(projection='3d')
        ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('f(x,t)')
        plt.savefig(f"Results\\Burgers2.png")
        plt.show()

        # %%

    def plot3D_Matrix(x, t, y, n):
        X, T = x, t
        F_xt = y
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
        fig.colorbar(cp)  # Add a colorbar to a plot
        ax.set_title('F(x,t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.savefig(f"Results\\Burgers1_pred_{n}.png")
        plt.show()
        ax = plt.axes(projection='3d')
        ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('f(x,t)')
        plt.show()
        plt.savefig(f"Results\\Burgers2_pred_{n}.png")

    def f_real(x, t):
        return torch.exp(-t) * (torch.sin(np.pi * x))

        # %% [markdown]
        # ### Neural Network

        # %%

    class Resnet(nn.Module):

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

    left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    left_Y = torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)
    # Boundary Conditions
    # Bottom Edge: x=min; tmin=<t=<max
    # First row # The [:,None] is to give it the right dimension
    bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))
    bottom_Y = torch.zeros(bottom_X.shape[0], 1)
    # Top Edge: x=max; 0=<t=<1
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
    X_train_Nf = lb + (ub - lb) * lhs(2, num_collocation_points)  # 2 as the inputs are x and t
    # Add the training poinst to the collocation points
    X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))

    # %%
    print("Original shapes for X and Y:", X.shape, y_real.shape)
    print("Boundary shapes for the edges:",
          left_X.shape, bottom_X.shape, top_X.shape)
    print("Available training data:", X_train.shape, Y_train.shape)
    print("Final training data:", X_train_Nu.shape, Y_train_Nu.shape)
    print("Total collocation points:", X_train_Nf.shape)

    # %% [markdown]

    # Split the data into training, validation, and testing sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=1234)

    # # Train Neural Network

    # %%
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
    optimizer = torch.optim.Adam(PINN.parameters(), lr=learning_rate, amsgrad=False)
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

    # %%
    # optimizer.step(PINN.closure)

    # Training loop
    for i in range(num_steps):
        if i == 0:
            print("Training Loss-----Validation Loss")
        loss = PINN.loss(X_train_Nu, Y_train_Nu, X_train_Nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % (num_steps / 10) == 0:
            with torch.no_grad():
                val_loss = PINN.lossBC(X_val, Y_val)
            print(loss.detach().cpu().numpy(), '---', val_loss.detach().cpu().numpy())

    # %% [markdown]
    # ### Plots

    # %%
    y1 = PINN(X_test)

    # %%
    x1 = X_test[:, 0]
    t1 = X_test[:, 1]
    [mse, rmse, mae, r2] = evaluate_metrics(x1, t1)
    np.save('NPY\\m6.npy', np.array([mse, rmse, mae, r2]))
    # %%
    arr_x1 = x1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
    arr_T1 = t1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
    arr_y1 = y1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
    arr_y_test = y_test.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()

    # %%
    plot3D_Matrix(arr_x1, arr_T1, arr_y1, 1)

    plot3D_Matrix(X, T, y_real, 2)

    # Evaluation on the testing dataset
    with torch.no_grad():
        test_loss = PINN.lossBC(X_test, Y_test)
        print("Final Testing Loss:", test_loss.detach().cpu().numpy())


if __name__ == '__main__':
    Pinn2()