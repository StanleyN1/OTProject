import matplotlib.pyplot as plt
import numpy as np
from stanley_appex.estimation import *
from stanley_appex.utils import *
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
        
# def plot_sampled(ts_data, xs_data, Pi_est, N_sample=10):
#     xs_sampled, idxs_sampled = sample_trajectory_xs(Pi_est, xs_data, N_sample=N_sample)

#     [plot_trajectory1d(xs_sampled, ts_data, i, show=False, scatter=False) for i in range(N_sample)]
#     plot_trajectory1d(xs_data, ts_data, 0, show=False, scatter=True, all=True, size=1)

def plot_kernel_coupling(idx, xs_data, ts_data, A, H):
    ticurr, tifuture = idx, idx+1
    t_curr, t_future = ts_data[ticurr], ts_data[tifuture]
    x = np.array(xs_data[ticurr])
    y = np.array(xs_data[tifuture])
    K_rect = kernel_vec(x, y, t_curr, t_future, A, H)

    N_curr = K_rect.shape[0]
    N_future = K_rect.shape[1]
    print(N_curr, N_future)
    mu = np.ones(N_curr) / N_curr
    nu = np.ones(N_future) / N_future
    coupling = OT_sinkhorn(mu, nu, K_rect, maxiter=1000, stopThr=1e-1, epsilon=np.inf)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(K_rect)
    axs[0].set_title("K_rect")
    axs[1].imshow(coupling)
    axs[1].set_title("Coupling")
    # fig.colorbar(im, cax=axs[1])
    plt.show()    

def plot_sampling(xs_data, ts_data, Pi_est, trajectories, branch_times_data, N_sample=10, plot_dim=1, reverse=False):
    xs_sampled, idxs_sampled = sample_trajectory_xs_rectangle(Pi_est, xs_data, N_sample=N_sample, reverse=reverse)
    N_traj = trajectories.shape[0]
    # 1d trajectories
    if plot_dim == 1:
        [plt.plot(ts_data, trajectories[i, :, 0], alpha=0.5) for i in range(N_traj)];
        [plt.plot(ts_data, xs_sampled[i, :, 0], color="black", alpha=0.3) for i in range(N_sample)];
        plt.xlabel('time')
        plt.ylabel('x')
        plt.title("Resampling Trajectories")
        for branch_time in branch_times_data:
            plt.axvline(ts_data[branch_time], color="red", alpha=0.3, label="branch times")
        plt.show()

    if plot_dim == 2:
        
        [plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.5, color="red") for i in range(N_traj)];
        [plt.plot(xs_sampled[i, :, 0], xs_sampled[i, :, 1], color="black", alpha=0.3) for i in range(N_sample)];
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Resampling Trajectories 2D")
        plt.show()
        
def matrix_estimates(process, downsample_rate=1, print_out=False):
    A_mle_all_data = A_mle(process.trajectories[:, :-1, :], process.ts)
    H_mle_all_data = H_mle(process.trajectories, process.ts, process.A)

    A_mle_downsampled = A_mle(process.trajectories[:, :-1:downsample_rate, :], process.ts[::downsample_rate])
    H_mle_downsampled = H_mle(process.trajectories[:, ::downsample_rate, :], process.ts[::downsample_rate], process.A)

    A_mle_downsampled_permute = A_mle(apply_permutation(process.trajectories[:, :-1:downsample_rate, :]), process.ts[::downsample_rate])

    if print_out:
        print("A_mle:\n", A_mle_all_data)
        print("A_mle_downsampled:\n", A_mle_downsampled)
        print("H_mle:\n", H_mle_all_data)
        print("H_mle downsampled:\n", H_mle_downsampled)
        print("A_mle downsampled permuted:\n", A_mle_downsampled_permute)
        print("A:\n", process.A)
        print("H:\n", process.H)
    return A_mle_all_data, A_mle_downsampled, H_mle_all_data, H_mle_downsampled, A_mle_downsampled_permute
    
def matrix_errors(process, As, Hs, downsample_rate=1, plot=False):
    '''
    outputs the error between the true A and H matrices and the estimated A and H matrices
    '''
    A_est = As[-1]
    A = process.A
    H_est = Hs[-1]
    H = process.H
    
    # A_mle_all_data = A_mle(process.trajectories[:, :-1, :], process.ts)
    # H_mle_all_data = H_mle(process.trajectories, process.ts, process.A)


    # A_mle_downsampled_permute = A_mle(apply_permutation(process.trajectories[:, :-1:downsample_rate, :]), process.ts[::downsample_rate])

    A_errors, H_errors = estimation_error(As, A, Hs, H)
    
    if plot:
        A_mle_downsampled = A_mle(process.trajectories[:, :-1:downsample_rate, :], process.ts[::downsample_rate])
        H_mle_downsampled = H_mle(process.trajectories[:, ::downsample_rate, :], process.ts[::downsample_rate], process.A)
        print("A mle error", np.linalg.norm(A_mle_downsampled - A, ord="fro") / np.linalg.norm(A, ord="fro"))
        print("H mle error", np.linalg.norm(H_mle_downsampled - H, ord="fro") / np.linalg.norm(H, ord="fro"))

        print("A algorithm error", np.linalg.norm(A_est - A, ord="fro"))
        print("H algorithm error", np.linalg.norm(H_est - H, ord="fro"))
        
        plt.plot((A_errors), label="A error")
        plt.plot((H_errors), label="H error")
        plt.ylabel('frobenius norm error')
        plt.xlabel('iteration')
        plt.legend()
        plt.show()
    
    return A_errors, H_errors