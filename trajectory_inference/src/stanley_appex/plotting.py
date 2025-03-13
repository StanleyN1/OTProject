import matplotlib.pyplot as plt
# from estimation import sample_trajectory_xs

def plot_time_matrix(A, rate=10):
    T = A.shape[2]
    for i in range(0, T, rate):
        plt.figure(figsize=(3, 3))
        # plt.subplot(1, T, i+1)
        plt.imshow(A[:, :, i])
        plt.colorbar()
        plt.show()
        

def plot_trajectory1d(xs, ts, idx, show=True, scatter=False, all=False, size=1):
    if not all:
        if not scatter:
            plt.plot(ts, xs[idx, :, 0], label=f"x {idx}")
            plt.plot(ts, xs[idx, :, 1], label=f"y {idx}")
            plt.legend()
        else:
            for i in range(xs.shape[1]):
                plt.scatter(ts[i], xs[idx, i, 0], color=f"blue", s=size)
                plt.scatter(ts[i], xs[idx, i, 1], color=f"blue", s=size)
    if all:
        if scatter:
            for i in range(xs.shape[1]):
                for j in range(xs.shape[0]):
                    plt.scatter(ts[i], xs[j, i, 0], color=f"blue", s=size)
                    plt.scatter(ts[i], xs[j, i, 1], color=f"blue", s=size)
    if show:
        plt.show()

def plot_trajectory2d(xs, ts, idx, show=True):
    if idx == -1:
        for i in range(xs.shape[0]):
            plt.plot(xs[i, :, 0], xs[i, :, 1], label=f"trajectory {i}")
    else:
        plt.plot(xs[idx, :, 0], xs[idx, :, 1], label=f"trajectory {idx}")
        
    if show:
        plt.show()
        
def plot_sampled(ts_data, xs_data, Pi_est, N_sample=10):
    xs_sampled, idxs_sampled = sample_trajectory_xs(Pi_est, xs_data, N_sample=N_sample)

    [plot_trajectory1d(xs_sampled, ts_data, i, show=False, scatter=False) for i in range(N_sample)]
    plot_trajectory1d(xs_data, ts_data, 0, show=False, scatter=True, all=True, size=1)