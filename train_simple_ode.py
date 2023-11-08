import time
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from Subfunctions.Models import Resnet
from Subfunctions.Metrics import evaluate_metrics

def target_function(x):
    """
    Define the target function. In this case, it's the sine function.
    """
    y = torch.sin(x)
    return y

def train_simple_ode():
    """
    Train a simple neural network to approximate a sine function and evaluate its performance.
    """
    global optimizer
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    if device == 'cuda':
        print(torch.cuda.get_device_name())

    num_steps = 20000
    learning_rate = 1e-3

    x = torch.linspace(0, 2 * np.pi, 500).view(-1, 1)
    y = target_function(x)
    print(x.shape, y.shape)

    fig, ax1 = plt.subplots()
    ax1.plot(x.detach().numpy(), y.detach().numpy(), color='blue', label='Real_Train')
    ax1.set_xlabel('x', color='black')
    ax1.set_ylabel('f(x)', color='black')
    ax1.tick_params(axis='y', color='black')
    ax1.legend(loc='upper left')
    plt.savefig("Results\\simple_ode.png")

    # Shuffle data indices
    indices = np.arange(500)
    np.random.shuffle(indices)

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    torch.manual_seed(123)
    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)
    layers = np.array([1, 50, 50, 20, 50, 50, 1])

    model = Resnet(layers, x_train, y_train)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    start_time = time.time()

    for i in range(num_steps):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = model.lossNN(x_train, y_train)  # use mean squared error
        loss.backward()
        optimizer.step()
        if i % (num_steps / 10) == 0:
            print(loss)

    print(model.lossNN(x_test.float().to(device), y_test.float().to(device)))

    y_pred = model(x_train)
    y_pred_test = model(x_test.float().to(device))

    _, indices2 = torch.sort(x_train.squeeze(1))
    _, indices3 = torch.sort(x_test.squeeze(1))

    x_train_plot = x_train[indices2]
    y_train_plot = y_train[indices2]
    y_pred_train_plot = y_pred[indices2]
    x_test_plot = x_test[indices3]
    y_test_plot = y_test[indices3]
    y_pred_test_plot = y_pred_test[indices3]

    # Visualize the results on the test data
    fig, ax1 = plt.subplots()
    ax1.plot(x_test_plot.detach().cpu().numpy(),
             y_test_plot.detach().cpu().numpy(), color='green', label='Real_Test')
    ax1.plot(x_test_plot.detach().cpu().numpy(),
             y_pred_test_plot.detach().cpu().numpy(), color='orange', label='Pred_Test')
    ax1.set_xlabel('x', color='black')
    ax1.set_ylabel('f(x)', color='black')
    ax1.tick_params(axis='y', color='black')
    ax1.legend(loc='upper left')
    plt.savefig("Results\\simple_ode_pred_test.png")

    # Evaluate and save metrics for the test data
    [mse_test, rmse_test, mae_test, r2_test] = evaluate_metrics(y_test_plot.detach().numpy(), y_pred_test_plot.detach().numpy())
    a_test = np.array([mse_test, rmse_test, mae_test, r2_test])
    np.save('NPY\\simple_ode_test.npy', a_test)

    [mse, rmse, mae, r2] = evaluate_metrics(y_train_plot.detach().numpy(), y_pred_train_plot.detach().numpy())
    a = np.array([mse, rmse, mae, r2])
    np.save('NPY\\simple_ode.npy', a)
    plt.show()

if __name__ == '__main__':
    train_simple_ode()
