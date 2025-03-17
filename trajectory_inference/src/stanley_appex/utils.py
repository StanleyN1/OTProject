import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
from scipy.optimize import linear_sum_assignment
from scipy.linalg import norm
from scipy.special import logsumexp

def normalize(x):
    return x/np.sum(x)

def relative_entropy(p, q):
    return np.sum(p * np.log(np.divide(p, q)))

def c(x, y, eps):
    return (x - y)**2 / (2 * eps**2)

def OT_sinkhorn(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=np.inf):
    '''
    Sinkhorn algorithm given Gibbs kernel K
    :param a: first marginal
    :param b: second marginal
    :param K: Gibbs kernel
    :param maxiter: max number of iteraetions
    :param stopThr: threshold for stopping
    :param epsilon: second stopping threshold
    :return:
    '''
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    for _ in range(maxiter):
        u_prev = u
        # Perform standard Sinkhorn update
        u = a / (K @ v)
        v = b / (K.T @ u)
        tmp = np.diag(u) @ K @ np.diag(v)

        # Check for convergence based on the error
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(u - u_prev) / np.linalg.norm(u_prev) < epsilon:
            break

    return tmp


# def OT_sinkhorn(mu, nu, K, maxiters = 1000):
#     u = np.ones_like(mu)
#     v = np.ones_like(nu)
#     for i in range(maxiters):
#         u = mu / (K @ v)
#         v = nu / (K.T @ u)
#     return np.diag(u) @ K @ np.diag(v) # returns coupling

def sinkhorn_log(a, b, K, maxiter=500, stopThr=1e-9, epsilon=1e-5):
    '''
    Logarithm-domain Sinkhorn algorithm given Gibbs kernel K
    :param a: first marginal
    :param b: second marginal
    :param K: Gibbs kernel K
    :param maxiter: max number of iterations
    :param stopThr: threshold for stopping
    :param epsilon: second stopping threshold
    :return:
    '''
    # Initialize log-domain variables
    log_K = np.log(K + 1e-300)  # Small constant to prevent log(0)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    log_u = np.zeros(K.shape[0])
    log_v = np.zeros(K.shape[1])

    for _ in range(maxiter):
        log_u_prev = log_u.copy()

        # Perform updates in the log domain using logsumexp
        log_u = log_a - logsumexp(log_K + log_v, axis=1)
        log_v = log_b - logsumexp(log_K.T + log_u[:, np.newaxis], axis=0)

        # Calculate the transport plan in the log domain
        log_tmp = log_K + log_u[:, np.newaxis] + log_v

        # Check for convergence based on the error
        tmp = np.exp(log_tmp)
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(log_u - log_u_prev) < epsilon:
            break

    return tmp

def expAt(A, t):
    return linalg.expm(A*t)

def cov_integrand(A, H, t):
    return expAt(A, t) @ H @ expAt(A, t).T

def cov(A, H, t0, tf, n_integrate=10):
    ts = np.linspace(t0, tf, n_integrate)
    cov = np.zeros(A.shape)
    for i in range(n_integrate-1):
        cov += cov_integrand(A, H, tf - ts[i]) * (ts[i+1] - ts[i])
    return cov

def cov_approx(A, H, t0, tf):
    return H * (tf - t0)

def mean(A, x, t0, tf):
    return expAt(A, tf - t0)@x

def mean_approx(A, x, t0, tf):
    return x + A@x*(tf - t0)

def downsample(xs, ts, downsample_rate):
    return xs[:, ::downsample_rate, :], ts[::downsample_rate]

def kernel_func(y, m, c):
    return np.exp(-0.5*(y - m).T @ np.linalg.inv(c) @ (y - m))

def kernel(x, y, t_curr, t_future, A, H):
    kernel_mean = mean(A, x, t_curr, t_future)
    kernel_cov = cov(A, H, t_curr, t_future)

    kernel_mean_approx = mean_approx(A, x, t_curr, t_future)
    kernel_cov_approx = cov_approx(A, H, t_curr, t_future)
    K = kernel_func(y, kernel_mean, kernel_cov)
    K_approx = kernel_func(y, kernel_mean_approx, kernel_cov_approx)
    return K, K_approx

def closest_permutation_matrix(A, norm_type='fro'):
    """
    Finds the closest permutation matrix to a given matrix A and computes the norm between them.

    Parameters:
        A (ndarray): Input square matrix of shape (n, n).
        norm_type (str): Type of matrix norm to compute ('fro' for Frobenius, '1', 'inf', etc.).

    Returns:
        P (ndarray): Closest permutation matrix to A.
        distance (float): Matrix norm between A and P.
    """
    # Ensure A is square
    n, m = A.shape
    if n != m:
        raise ValueError("Input matrix must be square.")

    # Step 1: Use the Hungarian algorithm to find the closest permutation
    cost_matrix = -A  # Convert to a cost minimization problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the permutation matrix P
    P = np.zeros_like(A)
    P[row_ind, col_ind] = 1

    # Step 2: Compute the distance between A and P using the specified norm
    distance = norm(A - P, ord=norm_type)

    return P, distance

def distance_from_permutation(Pi_est):
    N = Pi_est.shape[2] # time steps
    dist_from_permutation = np.zeros(N)
    N_traj = Pi_est.shape[0]
    for i in range(Pi_est.shape[2]):
        P, distance = closest_permutation_matrix(N_traj*Pi_est[:, :, i], norm_type='fro') # coupling matrix diagonal is 1/N_traj
        dist_from_permutation[i] = distance
    return dist_from_permutation

def estimation_error(As, A_true, Hs, H_true, norm="fro"):
    A_errors = [np.linalg.norm(A - A_true, ord=norm) for A in As]
    H_errors = [np.linalg.norm(H - H_true, ord=norm) for H in Hs]
    return A_errors, H_errors

def marginals(trajectories, downsample_rate=1):
    # trajectory : N_traj x Nt x d array
    # returns an N_traj x Nt x d array
    # note that N_traj depends on t though
    xs_data = []
    N_traj, Nt, d = trajectories.shape
    for j in range(0, Nt, downsample_rate):
        trajs = np.unique(trajectories[:, j, :], axis=0).tolist()
        xs_data.append(trajs)

    return xs_data

def apply_permutation(trajectories):
    # trajectory : N_traj x Nt x d array
    N_traj, Nt, d = trajectories.shape

    # Generate different permutation matrices for each time step
    P_t = np.zeros((Nt, N_traj, N_traj))
    for t in range(Nt):
        perm = np.random.permutation(N_traj)
        P_t[t, np.arange(N_traj), perm] = 1  # Create permutation matrix for time t

    # Apply the time-dependent permutation matrices
    trajs_permuted = np.einsum('tij,jtd->itd', P_t, trajectories)
    return trajs_permuted