import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class Resnet(nn.Module):
    def __init__(self, layers, x_train, y_train):
        super().__init__()
        self.layers = layers
        self.x_train = x_train
        self.y_train = y_train

        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.iter = 0

        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def lossNN(self, x, y):
        loss_val = self.loss_function(self.forward(x), y)
        return loss_val

    def closure(self):
        optimizer.zero_grad()
        loss = self.lossNN(self.x_train, self.y_train)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(loss)
        return loss


# Data splitting function
def split_data(x_data, y_data, test_size=0.3, random_state=42, batch_size=32):
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=random_state)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    num_samples = 1000
    num_features = 5
    x_data = np.random.rand(num_samples, num_features)
    y_data = np.random.rand(num_samples)

    train_loader, val_loader, test_loader = split_data(x_data, y_data)

