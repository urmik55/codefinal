# ----------------Importing Required Libraries ----------------#
import numpy as np
import pandas as pd
from keras import layers
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
from pyDOE import lhs  # Latin Hypercube Sampling


def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [mse, rmse, mae, r2]


# Generate synthetic data for a simple dynamic system (e.g., a mass-spring-damper system)
def generate_data(num_samples):
    t = np.linspace(0, 10, num_samples)
    x = np.sin(t)  # Simulated state of the system
    u = np.cos(t)  # Input signal (control input)
    return t, x, u


class ResidualModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.num_samples = 1000
        self.show = False
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def residual_block(self, x, num_filters):
        shortcut = x
        x = layers.Dense(num_filters, activation='relu')(x)
        x = layers.Dense(num_filters, activation=None)(x)  # No activation on the last layer
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # Define the loss function
    def custom_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def Model(self):
        input_layer = layers.Input(shape=(2,))
        x = layers.Dense(32, activation='relu')(input_layer)
        for _ in range(4):  # Create four residual blocks
            x = self.residual_block(x, 32)
        output_layer = layers.Dense(1)(x)  # Output layer for the derivative

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss=self.custom_loss)
        # t, x, u = generate_data(self.num_samples)

        # Train the neural network to approximate the derivative
        model.fit(self.x_train, self.y_train, epochs=10000)
        return model

        # [mse, rmse, mae, r2] = evaluate_metrics(x[1:], predicted_x)

    def __call__(self):
        return self.Model()


def main_sim():
    show = False
    num_samples = 1000

    t, x, u = generate_data(num_samples)
    # Prepare the training data
    input_data = np.column_stack((x[:-1], u[:-1]))  # Input: [x(t), u(t)]
    output_data = np.diff(x) / np.diff(t)  # True derivative: dx/dt

    # Use the trained model for predictions
    input_data_test = np.column_stack((x[1:], u[1:]))  # Input: [x(t+1), u(t+1)]

    model = (ResidualModel(input_data, output_data, input_data_test, input_data_test))()
    input_data_test = np.column_stack((model.predict(input_data_test).flatten(), u[1:]))  # Input: [x(t+1), u(t+1)]
    predicted_derivative = model.predict(input_data_test)
    # Integrate the predicted derivative to get an approximation of x(t)
    predicted_x = np.cumsum(predicted_derivative) * np.mean(np.diff(t)) + x[0]

    if show:
        plt.figure(figsize=(10, 5))
        plt.plot(t[1:], x[1:], label='True x(t)')
        plt.plot(t[1:], predicted_x, label='Predicted x(t)')
        plt.xlabel('Time')
        plt.ylabel('State (x)')
        plt.legend()
        plt.savefig("Results\\Simple_ode1_pred.png")
        plt.show()

    return predicted_x


# main_sim()

def parametric_ode_system(t, u, args):
    a1, b1, c1, d1, a2, b2, c2, d2 = \
        args[0], args[1], args[2], args[3], \
            args[4], args[5], args[6], args[7]
    x, y = u[0], u[1]
    dx_dt = a1 * x + b1 * y + c1 * tf.math.exp(-d1 * t)
    dy_dt = a2 * x + b2 * y + c2 * tf.math.exp(-d2 * t)
    return tf.stack([dx_dt, dy_dt])


def main_sim1():
    true_params = [1.11, 2.43, -3.66, 1.37, 2.89, -1.97, 4.58, 2.86]

    an_sol_x = lambda t: \
        -1.38778e-17 * np.exp(-8.99002 * t) - \
        2.77556e-17 * np.exp(-7.50002 * t) + \
        3.28757 * np.exp(-3.49501 * t) - \
        3.18949 * np.exp(-2.86 * t) + \
        0.258028 * np.exp(-1.37 * t) - \
        0.356108 * np.exp(2.63501 * t) + \
        4.44089e-16 * np.exp(3.27002 * t) + \
        1.11022e-16 * np.exp(4.76002 * t)

    an_sol_y = lambda t: \
        -6.23016 * np.exp(-3.49501 * t) + \
        5.21081 * np.exp(-2.86 * t) + \
        1.24284 * np.exp(-1.37 * t) - \
        0.223485 * np.exp(2.63501 * t) + \
        2.77556e-17 * np.exp(4.76002 * t)

    t_begin = 0.
    t_end = 1.5
    t_nsamples = 150
    t_space = np.linspace(t_begin, t_end, t_nsamples)

    dataset_outs = [tf.expand_dims(an_sol_x(t_space), axis=1),
                    tf.expand_dims(an_sol_y(t_space), axis=1)]

    t_space_tensor = tf.constant(t_space)
    x_init = tf.constant([0.], dtype=t_space_tensor.dtype)
    y_init = tf.constant([0.], dtype=t_space_tensor.dtype)
    u_init = tf.convert_to_tensor([x_init, y_init], dtype=t_space_tensor.dtype)
    args = [tf.Variable(initial_value=1., name='p' + str(i + 1), trainable=True, dtype=t_space_tensor.dtype)
            for i in range(0, 8)]

    learning_rate = 0.05
    epochs = 200
    x_train = tf.concat([dataset_outs[0], dataset_outs[1]], 1)

    model = (ResidualModel(x_train, t_space_tensor, dataset_outs[0], dataset_outs[1]))()
    input_data_test = np.column_stack((model.predict(x_train).flatten()))  # Input: [x(t+1), u(t+1)]
    return input_data_test


def parabolic_main():
    true_params = [1.11, 2.43, -3.66, 1.37, 2.89, -1.97, 4.58, 2.86]

    # Modify the analytical solutions to represent a parabolic system (x and y are functions of t)
    an_sol_x = lambda t: 0.5 * true_params[0] * t ** 2 + 0.5 * true_params[2] * np.exp(-true_params[3] * t)
    an_sol_y = lambda t: 0.5 * true_params[4] * t ** 2 + 0.5 * true_params[6] * np.exp(-true_params[7] * t)

    # Modify the initial conditions to match the parabolic system
    x_init = tf.constant([0.], dtype=tf.float64)
    y_init = tf.constant([0.], dtype=tf.float64)

    t_begin = 0.
    t_end = 1.5
    t_nsamples = 150
    t_space = np.linspace(t_begin, t_end, t_nsamples)

    dataset_outs = [tf.expand_dims(an_sol_x(t_space), axis=1), tf.expand_dims(an_sol_y(t_space), axis=1)]

    t_space_tensor = tf.constant(t_space)
    u_init = tf.convert_to_tensor([x_init, y_init], dtype=t_space_tensor.dtype)
    args = [tf.Variable(initial_value=val, trainable=True, dtype=tf.float64) for val in true_params]

    x_train = tf.concat([dataset_outs[0], dataset_outs[1]], 1)

    model = (ResidualModel(x_train, t_space_tensor, dataset_outs[0], dataset_outs[1]))()
    input_data_test = np.column_stack((model.predict(x_train).flatten()))  # Input: [x(t+1), u(t+1)]
    return input_data_test


def f(x):
    y = torch.sin(x)
    return y


def simple_ode1():
    x = torch.linspace(0, 2 * np.pi, 500).view(-1, 1)  # prepare to NN
    y = f(x)
    x = np.array(x)
    y = np.array(y)
    indices = np.arange(500)

    train_idx, test_idx, _, _ = train_test_split(
        indices, indices, test_size=0.2, random_state=42)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_train1 = np.concatenate([x_train, x_train], axis=1)
    x_test1 = np.concatenate([x_test, x_test], axis=1)

    model = (ResidualModel(x_train1, y_train, x_test1, y_test))()
    y_pred = model.predict(x_test1)
    print(evaluate_metrics(y_test, y_pred))

    return y_pred


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


def PINN21():
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
    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    # Create the mesh
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    # Evaluate real function
    y_real = f_real(X, T)
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
    X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and t
    # Add the training poinst to the collocation points
    X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))
    device = 'cpu'
    # Store tensors to GPU
    X_train_Nu = X_train_Nu.float().to(device)  # Training Points (BC)
    Y_train_Nu = Y_train_Nu.float().to(device)  # Training Points (BC)
    X_train_Nf = X_train_Nf.float().to(device)  # Collocation Points
    f_hat = torch.zeros(X_train_Nf.shape[0], 1).to(device)  # to minimize function

    X_test = x_test.float().to(device)  # the input dataset (complete)
    Y_test = y_test.float().to(device)  # the real solution

    model = (ResidualModel(np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)))()
    y_pred = model.predict(np.array(X_test))
    return y_pred


def f_BC(x):
    return 1 - torch.abs(x)


def f_real1(x):
    return torch.sin(np.pi * x)


def PDE(x):
    return -1 * (np.pi ** 2) * torch.sin(np.pi * x)


def Poison():
    min = -1
    max = 1
    total_points = 500
    # Nu: Number of training points (2 as we onlt have 2 boundaries), # Nf: Number of collocation points (Evaluate PDE)
    Nu = 2
    Nf = 250
    x = torch.linspace(min, max, total_points).view(-1, 1)  # prepare to NN
    y = f_real1(x)
    print(x.shape, y.shape)

    device = 'cpu'

    # Nu: Number of training point, # Nf: Number of colloction points
    # Set Boundary conditions x=min & x= max
    BC_1 = x[0, :]
    BC_2 = x[-1, :]
    # Total Tpaining points BC1+BC2
    all_train = torch.vstack([BC_1, BC_2])
    # Select Nu points
    idx = np.random.choice(all_train.shape[0], Nu, replace=False)
    x_BC = all_train[idx]

    # Latin Hypercube sampling for collocation points
    x_PDE = BC_1 + (BC_2 - BC_1) * lhs(1, Nf)
    x_PDE = torch.vstack((x_PDE, x_BC))

    # %%
    y_BC = f_BC(x_BC).to(device)

    # # Train Neural Network

    # %%
    # Store tensors to GPU
    torch.manual_seed(123)
    x_PDE = x_PDE.float().to(device)
    x_BC = x_BC.to(device)
    x_train = np.concatenate([x, x], axis=1)
    x_test = np.concatenate([x, x], axis=1)

    model = (ResidualModel(np.array(x_train), np.array(y), np.array(x_test), np.array(y)))()
    y_pred = model.predict(np.array(x_test))
    return y_pred


# Poison()
def PDE1():
    min = 0
    max = 2 * np.pi
    total_points = 500
    # Nu: Number of training points (2 as we onlt have 2 boundaries), # Nf: Number of collocation points (Evaluate PDE)
    Nu = 2
    Nf = 250
    device = 'cpu'
    # get the analytical solution over the full domain
    x = torch.linspace(min, max, total_points).view(-1, 1)  # prepare to NN
    y = f_BC(x)
    print(x.shape, y.shape)

    # Nu: Number of training point, # Nf: Number of colloction points
    # Set Boundary conditions:
    BC_1 = x[0, :]
    BC_2 = x[-1, :]
    # Total Training points BC1+BC2
    all_train = torch.vstack([BC_1, BC_2])
    # Select Nu points
    idx = np.random.choice(all_train.shape[0], Nu, replace=False)
    x_BC = all_train[idx]
    # Select Nf points
    # Latin Hypercube sampling for collocation points
    x_PDE = BC_1 + (BC_2 - BC_1) * lhs(1, Nf)
    x_PDE = torch.vstack((x_PDE, x_BC))

    torch.manual_seed(123)
    x_PDE = x_PDE.float().to(device)
    x_BC = x_BC.to(device)
    x_train = np.concatenate([x, x], axis=1)
    x_test = np.concatenate([x, x], axis=1)

    model = (ResidualModel(np.array(x_train), np.array(y), np.array(x_test), np.array(y)))()
    y_pred = model.predict(np.array(x_test))
    return y_pred


def PINN1():
    # To generate new data:
    x_min = -1
    x_max = 1
    t_min = 0
    t_max = 1
    total_points_x = 200
    total_points_t = 100
    device = 'cpu'
    # Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
    Nu = 100
    Nf = 10000
    x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, total_points_t).view(-1, 1)
    # Create the mesh
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    # Evaluate real function
    y_real = f_real(X, T)

    x_test = torch.hstack(
        (X.transpose(1, 0).flatten()[:, None], T.transpose(1, 0).flatten()[:, None]))
    # Colum major Flatten (so we transpose it)
    y_test = y_real.transpose(1, 0).flatten()[:, None]

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

    model = (ResidualModel(np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)))()
    y_pred = model.predict(np.array(X_test))

    return y_pred

def main1():
    PDE1()
    PINN1()
    PINN21()
    main_sim1()
    main_sim()
