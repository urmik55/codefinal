import time
import argparse
import os

import numpy as np

import tensorflow as tf

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', type=eval, default=False)
parser.set_defaults(viz=True)
args = parser.parse_args()

from tfdiffeq import odeint

tf.keras.backend.set_floatx('float64')

device = 'gpu:' + str(args.gpu) if tf.test.is_gpu_available() else 'cpu:0'

true_y0 = tf.convert_to_tensor([[0.5, 0.01]], dtype=tf.float64)
t = tf.linspace(0., 25., args.data_size)
true_A = tf.convert_to_tensor([[-0.1, 3.0], [-3.0, -0.1]], dtype=tf.float64)


class Lambda(tf.keras.Model):
    def call(self, t, y):
        return tf.matmul(y, true_A)


with tf.device(device):
    t1 = time.time()
    true_y = odeint(Lambda(), true_y0, t, method=args.method)
    t2 = time.time()
print(true_y)
print()
print("Time taken to compute solution : ", t2 - t1)


def get_batch():
    s = np.random.choice(
        np.arange(args.data_size - args.batch_time,
                  dtype=np.int64), args.batch_size,
        replace=False)

    temp_y = true_y.numpy()
    batch_y0 = tf.convert_to_tensor(temp_y[s])  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = tf.stack([temp_y[s + i] for i in range(args.batch_time)], axis=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        max_y, min_y = true_y.numpy().max(), true_y.numpy().min()

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-',
                     label='True trajectories')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--',
                     label='Predicted Trajectories')
        ax_traj.set_xlim(min(t.numpy()), max(t.numpy()))
        ax_traj.set_ylim(min_y, max_y)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(min_y, max_y)
        ax_phase.set_ylim(min_y, max_y)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[min_y:max_y:21j, min_y:max_y:21j]
        dydt = odefunc(0, tf.convert_to_tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(min_y, max_y)
        ax_vecfield.set_ylim(min_y, max_y)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ODEFunc, self).__init__(**kwargs)

        self.x = tf.keras.layers.Dense(50, activation='tanh',
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.y = tf.keras.layers.Dense(2,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

    def call(self, t, y):
        x = self.x(y)
        y = self.y(x)
        return y


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def circular_main():
    ii = 0
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    with tf.device(device):
        func = ODEFunc()

        lr = 1e-3
        optimizer = tf.keras.optimizers.RMSprop(lr)

        for itr in range(1, args.niters + 1):

            with tf.GradientTape() as tape:
                batch_y0, batch_t, batch_y = get_batch()
                pred_y = odeint(func, batch_y0, batch_t)
                loss = tf.reduce_mean(tf.abs(pred_y - batch_y))

            grads = tape.gradient(loss, func.variables)
            grad_vars = zip(grads, func.variables)

            optimizer.apply_gradients(grad_vars)

            time_meter.update(time.time() - end)
            loss_meter.update(loss.numpy())

            if itr % 10 == 0:
                pred_y = odeint(func, true_y0, t)
                loss = tf.reduce_mean(tf.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.numpy()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

            end = time.time()


if __name__ == '__main__':
    circular_main()
