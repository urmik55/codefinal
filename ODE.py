import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the neural network model
class FeedForwardNN(nn.Module):
    """
    FeedForward Neural Network for regression.
    """

    def __init__(self, layers):
        super(FeedForwardNN, self).__init__()

        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss(self, x, y):
        loss_val = self.loss_function(self.forward(x), y)
        return loss_val


# Define the function to approximate (sin(x))
def target_function(x):
    y = torch.sin(x)
    return y


# Split the data into training, validation, and test sets
x = torch.linspace(0, 2 * np.pi, 500).view(-1, 1)
y = target_function(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Convert data to the appropriate device
x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

# Define the neural network architecture and optimizer
layers = np.array([1, 50, 50, 20, 50, 50, 1])  # 5 hidden layers
model = FeedForwardNN(layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping parameters
early_stopping_patience = 100
best_val_loss = float('inf')
no_improvement_count = 0

# Train the neural network on the training set and monitor the validation set
num_epochs = 20000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    yh = model(x_train)
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % (num_epochs // 10) == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Monitor validation loss and implement early stopping
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            val_loss = model.loss(x_val, y_val)
            print(f'Validation Loss: {val_loss.item()}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print('Early stopping: No improvement for', early_stopping_patience, 'epochs.')
                break

# Evaluate the model on the test set
with torch.no_grad():
    test_loss = model.loss(x_test, y_test)
    print(f'Test Loss: {test_loss.item()}')

# Make predictions and plot the results
model.eval()
x_pred = torch.linspace(0, 2 * np.pi, 500).view(-1, 1).to(device)
y_pred = model(x_pred).detach().cpu().numpy()  # Detach and convert to NumPy

plt.plot(x_pred.cpu().numpy(), y_pred, label='Predicted', color='red')
plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='True', color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
