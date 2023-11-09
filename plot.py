import matplotlib.pyplot as plt
import numpy as np


def plot_picture(mif, loss, errors):
    plt.figure(figsize=(6, 5))
    plt.subplot(311)
    plt.plot(np.asarray(mif)[:, 0])
    plt.subplot(312)
    plt.plot(np.asarray(mif)[:, 1])
    plt.subplot(313)
    plt.plot(np.asarray(mif)[:, 2])

    plt.figure()
    plt.plot(np.asarray(loss)[:, 0])

    # plt.figure()
    # e = np.asarray(errors)
    # u = e.reshape(e.shape[0] * e.shape[1], e.shape[2])
    # plt.plot(u[:, 0])
    # plt.figure()
    # rmse = [np.sqrt(np.sum((u[s, :]) ** 2) / len(e)) for s in range(u.shape[0])]

    plt.figure()
    rmse = [np.sqrt(np.sum((errors[s, :, :]) ** 2) / (errors.shape[1] * errors.shape[2])) for s in
            range(errors.shape[0])]
    print(rmse)
    plt.plot(rmse)


if __name__ == '__main__':
    m = np.load('max_isometric_force.npy')
    l = np.load('loss.npy')
    e = np.load('error.npy')
    m0 = np.load('whole body model/Results1108/max_isometric_force.npy')
    l0 = np.load('whole body model/Results1108/loss.npy')
    e0 = np.load('whole body model/Results1108/error.npy')
    m1 = np.load('whole body model/Results1109/max_isometric_force.npy')
    l1 = np.load('whole body model/Results1109/loss.npy')
    e1 = np.load('whole body model/Results1109/error.npy')
    # print(m1[-1, 0], m1[-1, 1], m1[-1, 2])

    plot_picture(m, l, e)
    # plot_picture(m0, l0, e0)
    # plot_picture(m1, l1, e1)

    plt.show()