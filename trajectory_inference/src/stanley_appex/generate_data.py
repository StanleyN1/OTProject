import numpy as np
import matplotlib.pyplot as plt

def integrate_trajectory(x0, ts, A, H):
    A = np.array(A)
    H = np.array(H)
    dt = np.diff(ts)
    d = A.shape[1]
    xs = np.zeros((len(ts), d))
    xs[0] = x0

    for i in range(len(ts) - 1):
        xs[i + 1] = xs[i] + A@xs[i]*dt[i] + np.sqrt(dt[i])*np.random.multivariate_normal([0, 0], H)
    return xs

def many_trajectories(x0, A, H, N, ts):
    assert len(x0) == N
    xs = np.zeros((N, len(ts), A.shape[1]))
    for i in range(N):
        xs[i] = integrate_trajectory(x0[i], ts, A, H)
    return xs

class BranchingStochasticProcess:
    def __init__(self, A, G, X0, lambda_func, dt=0.01, T=1.0, N_traj=1):
        self.A = np.array(A)  # d x d drift matrix
        self.G = np.array(G)  # d x d diffusion matrix
        self.d = A.shape[0]
        self.X0 = np.array(X0)  # Initial position (d,)
        self.lambda_func = lambda_func  # Function defining branching rate
        self.dt = dt  # Time step
        self.T = T  # Total simulation time
        self.trajectories = []
        self.branch_times = []  # Store branching times
        self.N_traj = N_traj
        self.ts = np.arange(0, T, dt)
        self.Nt = len(self.ts)
        self.lineage = {}  # Store lineage of particles
    
    def simulate(self, X0):
        particles = X0  # Initial particles
        self.trajectories = [[X] for X in particles]  # Store all trajectories
        
        t = self.dt
        ti = 1
        while t < self.T:
            new_particles = []
            new_trajectories = []
            for i, X in enumerate(particles):
                
                dW = np.random.randn(len(X)) * np.sqrt(self.dt)
                X_new = X + self.A @ X * self.dt + self.G @ dW

                
                # Branching condition (Poisson process)
                if np.random.rand() < self.lambda_func(X) * self.dt:
                    new_particles.append(X_new)  # Keep the original
                    new_particles.append(X_new)  # Create a new branch at the same position
                    self.branch_times.append(ti + 1)  # Store branching time
                    
                    # Duplicate trajectory for new branch
                    new_trajectories.append(self.trajectories[i] + [X_new])
                    new_trajectories.append(self.trajectories[i] + [X_new])
                else:
                    new_particles.append(X_new)
                    new_trajectories.append(self.trajectories[i] + [X_new])
                
            particles = new_particles
            self.trajectories = new_trajectories
            t += self.dt
            ti += 1
        
        self.trajectories = np.array(self.trajectories)
        self.N_traj = self.trajectories.shape[0]
        self.time_marginals = self.marginals()
        
    def plot_trajectories(self, downsample=1):
        plt.figure(figsize=(8, 6))
        for traj in self.trajectories:
            traj = np.array(traj)
            plt.plot(traj[::downsample, 0], traj[::downsample, 1], alpha=0.7)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Branching Stochastic Process in 2D')
        plt.show()

    def marginals(self, downsample_rate=1):
        # returns an N_traj x Nt x d array
        # note that N_traj depends on t though
        xs_data = []
        for j in range(0, self.Nt, downsample_rate):
            trajs = np.unique(self.trajectories[:, j, :], axis=0).tolist()
            xs_data.append(trajs)

        return xs_data