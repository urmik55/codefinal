import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.optimizers as tfko
from tfdiffeq import odeint
from sklearn.model_selection import train_test_split  # Import for data shuffling and splitting
from Subfunctions import evaluate_metrics

def parabolic():
    def parametric_ode_system(t, u, args):
        a1, b1, c1, d1, a2, b2, c2, d2 = args
        x, y = u[0], u[1]
        dx_dt = a1 * x + b1 * y + c1 * tf.math.exp(-d1 * t)
        dy_dt = a2 * x + b2 * y + c2 * tf.math.exp(-d2 * t)
        return tf.stack([dx_dt, dy_dt])

    true_params = [1.11, 2.43, -3.66, 1.37, 2.89, -1.97, 4.58, 2.86]

    an_sol_x = lambda t: 0.5 * true_params[0] * t ** 2 + 0.5 * true_params[2] * np.exp(-true_params[3] * t)
    an_sol_y = lambda t: 0.5 * true_params[4] * t ** 2 + 0.5 * true_params[6] * np.exp(-true_params[7] * t)

    x_init = tf.constant([0.], dtype=tf.float64)
    y_init = tf.constant([0.], dtype=tf.float64)

    t_begin = 0.
    t_end = 1.5
    t_nsamples = 150
    t_space = np.linspace(t_begin, t_end, t_nsamples)

    # Shuffle and split the data
    x_data = np.expand_dims(an_sol_x(t_space), axis=1)
    y_data = np.expand_dims(an_sol_y(t_space), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    dataset_outs = [x_train, y_train]  # Use the shuffled and split data for training

    t_space_tensor = tf.constant(t_space)
    u_init = tf.convert_to_tensor([x_init, y_init], dtype=t_space_tensor.dtype)
    args = [tf.Variable(initial_value=val, trainable=True, dtype=tf.float64) for val in true_params]

    learning_rate = 0.05
    epochs = 100
    optimizer = tfko.Adam(learning_rate=learning_rate)

    def net():
        return odeint(lambda ts, u0: parametric_ode_system(ts, u0, args), u_init, t_space_tensor)

    def loss_func(num_sol):
        # Take the first len(dataset_outs[0]) elements from num_sol for subtraction
        num_sol_subset = num_sol[:len(dataset_outs[0])]
        return tf.reduce_sum(tf.square(dataset_outs[0] - num_sol_subset[:, 0])) + \
               tf.reduce_sum(tf.square(dataset_outs[1] - num_sol_subset[:, 1]))

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            num_sol = net()
            loss_value = loss_func(num_sol)
        print("Epoch:", epoch, " loss:", loss_value.numpy())
        grads = tape.gradient(loss_value, args)
        optimizer.apply_gradients(zip(grads, args))

    print("Learned parameters:", [args[i].numpy() for i in range(0, len(args))])

    num_sol = net()
    x_num_sol = num_sol[:, 0].numpy()
    y_num_sol = num_sol[:, 1].numpy()

    x_an_sol = an_sol_x(t_space)
    y_an_sol = an_sol_y(t_space)

    [mse, rmse, mae, r2] = evaluate_metrics(y_num_sol, y_an_sol)
    np.save('NPY\\m7.npy', np.array([mse, rmse, mae, r2]))

    plt.figure()
    plt.plot(t_space, an_sol_x(t_space), '--', linewidth=2, label='analytical x')
    plt.plot(t_space, an_sol_y(t_space), '--', linewidth=2, label='analytical y')
    plt.plot(t_space, x_num_sol, linewidth=1, label='numerical x')
    plt.plot(t_space, y_num_sol, linewidth=1, label='numerical y')
    plt.title('Neural ODEs to fit params')
    plt.xlabel('t')
    plt.legend()
    plt.savefig("Results\\Parabolic_pred.png")
    plt.show()

if __name__ == "__main__":
    parabolic()
