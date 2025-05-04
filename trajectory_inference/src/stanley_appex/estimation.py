import numpy as np
from scipy.linalg import expm
from stanley_appex.utils import kernel, OT_sinkhorn, cov, cov_approx, sinkhorn_log
import matplotlib.pyplot as plt

def pairwise_kernel(xs, ts, t_idx, A, H):
    N_traj = xs.shape[0]
    K = np.zeros((N_traj, N_traj))
    K_approx = np.zeros((N_traj, N_traj))
    for i in range(N_traj):
        for j in range(N_traj):
            t_curr = ts[t_idx]
            t_future = ts[t_idx + 1]
            x = xs[i, t_idx, :]
            y = xs[j, t_idx + 1, :]

            K[i,j], K_approx[i,j] = kernel(x, y, t_curr, t_future, A, H)

    return K, K_approx

def time_pairwise_kernel(xs, ts, A, H):
    N_traj = xs.shape[0]
    K = np.zeros((N_traj, N_traj, len(ts) - 1))
    K_approx = np.zeros((N_traj, N_traj, len(ts) - 1))

    for t_idx in range(len(ts) - 1):
        K_, K_approx_ = pairwise_kernel(xs, ts, t_idx, A, H)
        K[:, :, t_idx] = K_
        K_approx[:, :, t_idx] = K_approx_

    return K, K_approx

def kernel_approx_vec(x, y, t_curr, t_future, A, H):
    # kernel between (x, t_curr) and (y, t_future)
    m = x + x@A.T*(t_future - t_curr) # e^(Adt) x
    d = y[:, None] - m[None] # y - e^(Adt) x
    c = cov_approx(A, H, t_curr, t_future) # cov
    
    return np.exp(-np.einsum('ijk,kl,ijl->ji', d, np.linalg.inv(c), d)) # returns N x N kernel

def kernel_vec(x, y, t_curr, t_future, A, H):
    # matrix kernel between (x, t_curr) and (y, t_future)
    m = (expm(A*(t_future - t_curr))@(x.T)).T # e^(Adt) x
    d = y[:, None] - m[None] # y - e^(Adt) x
    c = cov(A, H, t_curr, t_future) # cov
    return np.exp(-0.5*np.einsum('ijk, ijl, kl -> ji', d, d, np.linalg.inv(c))) 

# def kernel_vec(x, y, t_curr, t_future, A, H):
#     # matrix kernel between (x, t_curr) and (y, t_future)
#     m = x@expm(A.T*(t_future - t_curr)) # e^(Adt) x
#     d = y[:, None] - m[None] # y - e^(Adt) x
#     c = cov(A, H, t_curr, t_future) # cov
    
#     return np.exp(-np.einsum('ijk,kl,ijl->ji', d, np.linalg.inv(c), d)) # returns N x N kernel

def time_pairwise_kernel_rectangle(xs, ts, A, H, only_approx=False):

    assert len(xs) == len(ts), "length of xs and ts must be the same"
    K = []
    K_approx = []
    for i in range(len(ts) - 1):
        x = np.array(xs[i])
        y = np.array(xs[i + 1])
        t_curr = ts[i]
        t_future = ts[i + 1]
        if only_approx:
            K_approx_ = kernel_approx_vec(x, y, t_curr, t_future, A, H)
            K_ = K_approx_
        else:
            K_ = kernel_vec(x, y, t_curr, t_future, A, H)
            K_approx_ = kernel_approx_vec(x, y, t_curr, t_future, A, H)
        K.append(K_)
        K_approx.append(K_approx_)

    return K, K_approx

## MLE Estimation
def A_mle(xs, ts, ridge_lambda=0):
    # xs: N_traj x N x d
    # ts: N
    assert ts.shape[0] == xs.shape[1]
    dt = np.diff(ts)[0]
    N_traj = xs.shape[0]
    d = xs.shape[2]
    N = len(ts)
    # dt_xs = np.einsum('ijk,j->ijk', xs, dt)
    X_X = np.einsum('ijk,ijl -> kl', xs*dt, xs)

    dxs = np.diff(xs, axis=1)
    DX_X = np.einsum('ijk,ijl -> kl', dxs, xs[:, :-1, :])

    A = DX_X @ np.linalg.inv(X_X + ridge_lambda*np.eye(d))
    return A

def H_mle(xs, ts, A):
    # xs: N_traj x N x d
    # ts: N
    dt = np.diff(ts)[0]
    N_traj = xs.shape[0]
    d = xs.shape[2]
    N = len(ts)
    T = ts[-1]
    dxs = np.diff(xs, axis=1)

    AX = np.einsum("lk,ijk -> ijl", A, xs[:, :-1, :])
    DX_AXdt = dxs - dt*AX

    H = (1 / (N_traj*T)) * np.einsum('ijk,ijl -> kl', DX_AXdt, DX_AXdt)
    return H

def OT_time_kernel(xs, ts, A, H, maxiters = 1000):
    K, K_approx = time_pairwise_kernel(xs, ts, A, H)
    N_traj = K.shape[0]
    Pi = np.zeros_like(K)
    for t_idx in range(len(ts) - 1):
        mu = np.ones(N_traj) / N_traj
        nu = np.ones(N_traj) / N_traj
        pi = OT_sinkhorn(mu, nu, K[:, :, t_idx], maxiters=maxiters)
        Pi[:, :, t_idx] = pi
    return Pi, K, K_approx

def OT_time_kernel_rectangle(xs, ts, A, H, maxiters = 1000):
    # K: N x N_traj(t_curr) x N_traj(t_future)
    # Pi: N x N_traj(t_curr) x N_traj(t_future)
    assert len(xs) == len(ts), "length of xs and ts must be the same"
    assert len(xs[0][0]) == len(A), 'dimension of xs and A must be the same'
    assert len(xs[0][0]) == len(H), 'dimension of xs and H must be the same'
    Pi = []
    K, K_approx = time_pairwise_kernel_rectangle(xs, ts, A, H, only_approx=False)
    for t_idx in range(len(ts) - 1):
        N_curr = K[t_idx].shape[0]
        N_future = K[t_idx].shape[1]
    
        mu = np.ones(N_curr) / N_curr
        nu = np.ones(N_future) / N_future
        
        pi = OT_sinkhorn(mu, nu, K[t_idx])
        # pi = sinkhorn_log(mu, nu, K[t_idx], maxiter=maxiters)
        Pi.append(pi)
    return Pi, K, K_approx

def OT_time_kernel_rectangle_unnormalized(xs, ts, A, H, maxiters = 1000):
    # K: N x N_traj(t_curr) x N_traj(t_future)
    # Pi: N x N_traj(t_curr) x N_traj(t_future)
    Pi = []
    K, K_approx = time_pairwise_kernel_rectangle(xs, ts, A, H, only_approx=False)
    for t_idx in range(len(ts) - 1):
        N_curr = K[t_idx].shape[0]
        N_future = K[t_idx].shape[1]
    
        mu = np.ones(N_curr) # / N_curr
        nu = np.ones(N_future) # / N_future
        
        pi = OT_sinkhorn(mu, nu, K[t_idx])
        # pi = sinkhorn_log(mu, nu, K[t_idx], maxiter=maxiters)
        Pi.append(pi)
    return Pi, K, K_approx

def normalize_column(column):
    return column / np.sum(column)

def sample_trajectory_idxs(Pi, N_sample):
    # Pi: N_traj, N_traj, N
    N_traj = Pi.shape[0] # assuming constant number of trajectories
    N = Pi.shape[2] + 1
    idxs_sampled = np.zeros((N_sample, N), dtype=int)
    
    for i in range(N_sample):
        sample_idx = np.random.choice(N_traj) # uniformly choose a random trajectory
        idxs_sampled[i, 0] = sample_idx
        
        for t_idx in range(1, N):
            pi_next_given_curr = normalize_column(Pi[:, sample_idx, t_idx - 1]) # conditional p_{i+1 | i}
            sample_idx = np.random.choice(N_traj, p=pi_next_given_curr)
            idxs_sampled[i, t_idx] = sample_idx
    return idxs_sampled



def sample_trajectory_idxs_rectangle(Pi, N_sample):
    # idxs_sampled: N_sample x N
    N = len(Pi) + 1
    idxs_sampled = np.zeros((N_sample, N), dtype=int)
    
    for i in range(N_sample):
        N_traj = Pi[0].shape[0] # assuming constant number of trajectories
        sample_idx = np.random.choice(N_traj) # uniformly choose a random trajectory
        idxs_sampled[i, 0] = sample_idx
        
        for t_idx in range(1, N):
            # N_traj = Pi[t_idx - 1].shape[0] previous
            N_traj = Pi[t_idx - 1].shape[1] # future
            
            # pi_next_given_curr = normalize_column(Pi[t_idx - 1][:, sample_idx]) # conditional p_{i+1 | i} previous
            pi_next_given_curr = normalize_column(Pi[t_idx - 1][sample_idx, :]) # conditional p_{i+1 | i}
            # plt.imshow(Pi[t_idx - 1])
            # print("time index", t_idx - 1)
            # plt.show()
            sample_idx = np.random.choice(N_traj, p=pi_next_given_curr)
            idxs_sampled[i, t_idx] = sample_idx
            
    return idxs_sampled

def sample_trajectory_idxs_rectangle_reverse(Pi, N_sample):
    # idxs_sampled: N_sample x N
    N = len(Pi)
    idxs_sampled = np.zeros((N_sample, N+1), dtype=int)
    
    for i in range(N_sample):
        N_traj = Pi[N-1].shape[0] # assuming constant number of trajectories
        sample_idx = np.random.choice(N_traj) # uniformly choose a random trajectory
        idxs_sampled[i, N] = sample_idx
        
        for t_idx in range(N-1, -1, -1):
            N_traj = Pi[t_idx].shape[0] # previous
            # N_traj = Pi[t_idx - 1].shape[1] # future
            
            # pi_next_given_curr = normalize_column(Pi[t_idx - 1][:, sample_idx]) # conditional p_{i+1 | i} previous
            pi_next_given_curr = normalize_column((Pi[t_idx].T)[sample_idx, :]) # conditional p_{i+1 | i}
            sample_idx = np.random.choice(N_traj, p=pi_next_given_curr)
            idxs_sampled[i, t_idx] = sample_idx
            
    return idxs_sampled

def index_trajectory(xs, idxs_sampled):
    # xs: N_traj x N x d
    # idxs_sampled: N_sample x N
    assert idxs_sampled.shape[1] == xs.shape[1] # same number of time steps
    N = idxs_sampled.shape[1]
    N_sample = idxs_sampled.shape[0]
    d = xs[0].shape[1]
    xs_sampled = np.zeros((N_sample, N, d))
    for i in range(N):
        xs_sampled[:, i, :] = xs[i][idxs_sampled[:, i], :]
    return xs_sampled

def index_trajectory_rectangle(xs, idxs_sampled):
    # xs: N x N_traj x d
    # idxs_sampled: N_sample x N
    assert idxs_sampled.shape[1] == len(xs), f"length of sampled traj: {idxs_sampled.shape[1]}, length of data traj: {len(xs)}" # same number of time steps
    N = idxs_sampled.shape[1]
    N_sample = idxs_sampled.shape[0]
    d = len(xs[0][0])
    xs_sampled = np.zeros((N_sample, N, d))
    for i in range(N):
        for sample_traj_idx, traj_idx in enumerate(idxs_sampled[:, i]): # idx
            xs_sampled[sample_traj_idx, i, :] = xs[i][traj_idx]
    return xs_sampled

def sample_trajectory_xs(Pi, xs, N_sample):
    idxs_sampled = sample_trajectory_idxs(Pi, N_sample)
    return index_trajectory(xs, idxs_sampled), idxs_sampled

def sample_trajectory_xs_rectangle(Pi, xs, N_sample, reverse=False):
    if not reverse:
        idxs_sampled = sample_trajectory_idxs_rectangle(Pi, N_sample)
    else:
        idxs_sampled = sample_trajectory_idxs_rectangle_reverse(Pi, N_sample)
    return index_trajectory_rectangle(xs, idxs_sampled), idxs_sampled

def appex(xs_data, ts_data, A, H, N_sample, tol = 1e-5, maxiters = 100):
    d = xs_data.shape[2]
    As = [np.zeros((d, d)), A] # collection of A matrices
    Hs = [np.zeros((d, d)), H] #
    Pis = []
    i = 0
    while i < maxiters:
        # if np.linalg.norm(As[-1] - As[-2]) < tol:
        #     print(f"tolerance reached at iteration {i}")
        #     break
        Pi, K, K_approx = OT_time_kernel(xs_data, ts_data, As[-1], Hs[-1], maxiters = 100)
        xs_sampled, idxs_sampled = sample_trajectory_xs(Pi, xs_data, N_sample = N_sample)
        A_mle_ = A_mle(xs_sampled, ts_data)
        H_mle_ = H_mle(xs_sampled, ts_data, A_mle_)
        As.append(A_mle_)
        Hs.append(H_mle_)
        Pis.append(Pi)
        i += 1
        print(i)
    return As, Hs, Pis

def appex_rectangle(xs_data, ts_data, A_guess, H_guess, N_sample, tol = 1e-5, maxiters = 100, print_out = 100, save_coupling = False, ridge_lambda=0.0, reverse=False):
    assert len(xs_data) == len(ts_data), "length of xs and ts must be the same"
    assert len(xs_data[0][0]) == len(A_guess), 'dimension of xs and A must be the same'
    assert len(xs_data[0][0]) == len(H_guess), 'dimension of xs and H must be the same'
    
    d = A_guess.shape[0]
    As = [np.ones((d, d)), A_guess] # collection of A matrices
    Hs = [np.ones((d, d)), H_guess] #
    Pis = []
    i = 0
    
    while i < maxiters:
        running_tol = np.linalg.norm(As[-1] - As[-2])
        # if running_tol < tol:
        #     print(f"tolerance {running_tol} reached at iteration {i}")
        #     break
        if i % print_out == 0:
            print(f"iteration {i}, running tolerance {running_tol}")
        # Pi, K, K_approx = OT_time_kernel(xs_data, ts_data, As[-1], Hs[-1], maxiters = 100)

        Pi, K, K_approx = OT_time_kernel_rectangle(xs_data, ts_data, As[-1], Hs[-1], maxiters = 500)
        # Pi, K, K_approx = OT_time_kernel_rectangle_unnormalized(xs_data, ts_data, As[-1], Hs[-1], maxiters = 500)
        # xs_sampled, idxs_sampled = sample_trajectory_xs(Pi, xs_data, N_sample = N_sample)
        xs_sampled, idxs_sampled = sample_trajectory_xs_rectangle(Pi, xs_data, N_sample = N_sample, reverse=reverse)
        
        A_mle_ = A_mle(xs_sampled, ts_data, ridge_lambda=ridge_lambda)
        H_mle_ = H_mle(xs_sampled, ts_data, A_mle_)
        As.append(A_mle_)
        Hs.append(H_mle_)
        if save_coupling:
            Pis.append(Pi)
        i += 1
    return As, Hs, Pis

def run_appex(hfile, A_guess = 0.01*np.eye(2), H_guess = np.eye(2), maxiters=30, return_errors=False):
    """currently only for no branching data"""
    # xs_data: (N, N_traj, d)
    # ts_data: (N,)
    downsample_rate = hfile['xs_data'].attrs['downsample_rate']
    N_traj = hfile['xs_data'].attrs['N_traj']

    ts_data = hfile['ts_data'][:]
    xs_data = hfile['xs_data'][:].transpose(1, 0, 2)

    ridge_lambda = 0.0
    print(xs_data.shape, ts_data.shape)
    As, Hs, Pis = appex_rectangle(xs_data, ts_data, A_guess, H_guess, N_sample=N_traj*10, ridge_lambda=ridge_lambda, tol=1e-5, maxiters=maxiters, print_out=10, save_coupling=True, reverse=False)

    A_est = As[-1]
    H_est = Hs[-1]
    Pi_est = Pis[-1]
    print("A_est:\n", A_est)
    print("H_est:\n", H_est)
    
    A_true = hfile['xs_data'].attrs['A']
    G_true = hfile['xs_data'].attrs['G']
    H_true = G_true @ G_true.T
    
    if return_errors:
        A_errors = [np.linalg.norm(A - A_true, ord="fro") for A in As]
        H_errors = [np.linalg.norm(H - H_true, ord="fro") for H in Hs]
    
        return A_est, H_est, A_errors, H_errors
    else:
        return A_est, H_est