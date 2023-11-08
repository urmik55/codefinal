import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
from sklearn.model_selection import train_test_split
from pyDOE import lhs  # Latin Hypercube Sampling
import time

def PDE():
    """
    Solves a Partial Differential Equation (PDE) using a neural network.

    This function sets up and trains a neural network to solve a PDE with boundary conditions.

    Returns:
        None
    """
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)

    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name()}")

    # Define tuning parameters
    num_steps = 5000
    learning_rate = 1e-3
    hidden_layers = np.array([1, 50, 50, 20, 50, 50, 1])  # 5 hidden layers
    x_min = 0
    x_max = 2 * np.pi
    total_points = 500
    num_boundary_points = 2
    num_collocation_points = 250

    def boundary_condition(x):
        """
        Boundary condition function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Value satisfying the boundary condition.
        """
        return torch.sin(x)

    def pde_equation(x):
        """
        PDE equation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: PDE equation result.
        """
        return torch.cos(x)

    class ResNet(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.activation = nn.Tanh()
            self.loss_function = nn.MSELoss(reduction='mean')
            self.linears = nn.ModuleList(
                [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            self.iteration = 0

            for i in range(len(layers) - 1):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
                nn.init.zeros_(self.linears[i].bias.data)

        def forward(self, x):
            a = x.float()
            for i in range(len(self.linears) - 1):
                z = self.linears[i](a)
                a = self.activation(z)
            a = self.linears[-1](a)
            return a

        def loss_boundary_condition(self, x_boundary):
            loss_bc = self.loss_function(self.forward(x_boundary), boundary_condition(x_boundary))
            return loss_bc

        def loss_pde(self, x_collocation):
            g = x_collocation.clone()
            g.requires_grad = True
            f = self.forward(g)
            f_x = autograd.grad(f, g, torch.ones([x_collocation.shape[0], 1]).to(device),
                               retain_graph=True, create_graph=True)[0]
            loss_pde = self.loss_function(f_x, pde_equation(g))
            return loss_pde

        def total_loss(self, x_boundary, x_collocation):
            loss_bc = self.loss_boundary_condition(x_boundary)
            loss_pde = self.loss_pde(x_collocation)
            return loss_bc + loss_pde

        def closure(self):
            optimizer.zero_grad()
            loss = self.total_loss(x_train, y_train)
            loss.backward()
            self.iteration += 1
            if self.iteration % 100 == 0:
                print(f'Iteration {self.iteration}, Loss: {loss}')
            return loss

    # Generate the analytical solution over the full domain
    x = torch.linspace(x_min, x_max, total_points).view(-1, 1)
    y = boundary_condition(x)

    fig, ax1 = plt.subplots()
    ax1.plot(x.detach().numpy(), y.detach().numpy(), color='blue', label='Real_Train')
    ax1.set_xlabel('x', color='black')
    ax1.set_ylabel('f(x)', color='black')
    ax1.tick_params(axis='y', color='black')
    ax1.legend(loc='upper left')
    plt.savefig("Results/PDE.png")

    # Set boundary conditions
    boundary_1 = x[0, :]
    boundary_2 = x[-1, :]
    all_boundary_points = torch.vstack([boundary_1, boundary_2])
    boundary_indices = np.random.choice(all_boundary_points.shape[0], num_boundary_points, replace=False)
    x_boundary = all_boundary_points[boundary_indices]

    # Generate collocation points using Latin Hypercube sampling
    collocation_points = boundary_1 + (boundary_2 - boundary_1) * lhs(1, num_collocation_points)
    x_collocation = torch.vstack((collocation_points, x_boundary))

    torch.manual_seed(123)
    x_collocation = x_collocation.float().to(device)
    x_boundary = x_boundary.to(device)

    # Split the collocation points
    train_ratio = 0.8
    num_collocation_train = int(train_ratio * num_collocation_points)
    num_collocation_val = num_collocation_points - num_collocation_train

    x_collocation_train = x_collocation[:num_collocation_train]
    x_collocation_val = x_collocation[num_collocation_train:]

    # Concatenate the boundary points with the training set
    x_train = torch.vstack([x_boundary, x_collocation_train])
    y_train = torch.vstack([boundary_condition(x_boundary), boundary_condition(x_collocation_train)])

    # Validation set
    x_val = x_collocation_val
    y_val = boundary_condition(x_collocation_val)

    # Create the model
    model = ResNet(hidden_layers)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    for i in range(num_steps):
        yh = model(x_collocation)
        loss = model.total_loss(x_collocation, x_boundary)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % (num_steps / 10) == 0:
            print(f'Iteration {i}, Loss: {loss}')

    # Function
    yh = model(x.to(device))
    y = boundary_condition(x)

    # Error
    loss_bc = model.loss_boundary_condition(x.to(device))
    print(f'Boundary Condition Loss: {loss_bc}')

    # Derivative
    g = x.to(device)
    g = g.clone()
    g.requires_grad = True
    f = model(g)
    f_x = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]

    # Detach from GPU
    y_plot = y.detach().numpy()
    yh_plot = yh.detach().cpu().numpy()
    f_x_plot = f_x.detach().cpu().numpy()

    # Plot
    fig, ax1 = plt.subplots()
    ax1.plot(x, y_plot, color='blue', label='Real')
    ax1.plot(x, yh_plot, color='red', label='Predicted')
    ax1.plot(x, f_x_plot, color='green', label='Derivative')
    ax1.set_xlabel('x', color='black')
    ax1.set_ylabel('f(x)', color='black')
    ax1.tick_params(axis='y', color='black')
    ax1.legend(loc='upper left')
    plt.savefig("Results/PDE_pred.png")
    plt.show()

if __name__ == "__main__":
    PDE()
