import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from Metrics import evaluate_metrics

def simpleode1():
    """
    Simulate and train a neural network for approximating the derivative of a dynamic system.
    """
    # Generate synthetic data for a simple dynamic system (e.g., a mass-spring-damper system)
    def generate_data(num_samples):
        t = np.linspace(0, 10, num_samples)
        x = np.sin(t)  # Simulated state of the system
        u = np.cos(t)  # Input signal (control input)
        return t, x, u

    # Generate synthetic data
    num_samples = 1000
    t, x, u = generate_data(num_samples)

    # Define a ResNet-like neural network for approximating the derivative
    def residual_block(x, num_filters):
        shortcut = x
        x = layers.Dense(num_filters, activation='relu')(x)
        x = layers.Dense(num_filters, activation=None)(x)  # No activation on the last layer
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    input_layer = layers.Input(shape=(2,))
    x = layers.Dense(32, activation='relu')(input_layer)
    for _ in range(4):  # Create four residual blocks
        x = residual_block(x, 32)
    output_layer = layers.Dense(1)(x)  # Output layer for the derivative

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Define the loss function
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # Compile the model
    model.compile(optimizer='adam', loss=custom_loss)

    # Prepare the training data
    t, x, u = generate_data(num_samples)
    input_data = np.column_stack((x, u))
    output_data = np.diff(x) / np.diff(t)  # True derivative: dx/dt

    test_size = int(0.7 * len(input_data))
    val_size = int(0.8 * len(input_data))

    input_data_train = input_data[:test_size]
    output_data_train = output_data[:test_size]

    input_data_val = input_data[test_size:val_size]
    output_data_val = output_data[test_size:val_size]

    input_data_test = input_data[val_size:]
    output_data_test = output_data[val_size:]

    # Train the neural network to approximate the derivative
    model.fit(input_data_train, output_data_train, epochs=1000, validation_data=(input_data_val, output_data_val))

    # Use the trained model for predictions
    predicted_derivative = model.predict(input_data_test)

    # Integrate the predicted derivative to get an approximation of x(t)
    [mse, rmse, mae, r2] = evaluate_metrics(output_data_test, predicted_derivative[:-1])
    np.save('NPY\\m1.npy', np.array([mse, rmse, mae, r2]))

    # Plot the true and predicted system dynamics
    plt.figure(figsize=(10, 5))
    plt.plot(t[val_size + 1:], x[330:529], label='True x(t)')
    plt.plot(t[val_size:], predicted_derivative[:, 0], label='Predicted x(t)')
    plt.xlabel('Time')
    plt.ylabel('State (x)')
    plt.legend()
    plt.savefig("Results\\Simple_ode1_pred.png")
    plt.show()

if __name__ == "__main__":
    simpleode1()
