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
    def __init__(self, A, G, dt=0.01, Nt=15000, T=1.0, N_traj=1):
        self.A = np.array(A)  # d x d drift matrix
        self.G = np.array(G)  # d x d diffusion matrix
        self.H = self.G@self.G.T
        self.d = A.shape[0]
        # self.X0 = np.array(X0)  # Initial position (d,)
        # self.lambda_func = lambda_func  # Function defining branching rate
        self.dt = dt  # Time step
        self.Nt = int(Nt)  # Number of time steps
        self.trajectories = []
        self.branch_times = []  # Store branching times
        self.N_traj = N_traj
        self.ts = np.arange(0, Nt) * dt
        self.T = self.ts[-1]  # Total simulation time
        print(len(self.ts))
        assert len(self.ts) == self.Nt, "Length of ts must be equal to Nt"
        self.Nt = len(self.ts)
        self.lineage = {}  # Store lineage of particles
    
    def simulate(self, X0, growth_rate=0.0):
        if not callable(growth_rate):
            growth_rate_func = lambda x: growth_rate
            self.growth_rate = growth_rate
        else:
            growth_rate_func = growth_rate
            
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
                if np.random.rand() < growth_rate_func(X) * self.dt:
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
        
    def plot_trajectories(self, downsample=1, dim=2):
        plt.figure(figsize=(4, 4))
        if dim == 0:
            for traj in self.trajectories:
                traj = np.array(traj)
                plt.plot(traj[::downsample, 0], alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('X')
            plt.title('Branching Stochastic Process in 1D')
            plt.show()
            
        if dim == 1:
            for traj in self.trajectories:
                traj = np.array(traj)
                plt.plot(traj[::downsample, 1], alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('X')
            plt.title('Branching Stochastic Process in 1D')
            plt.show()
            
        if dim == 2:
            for traj in self.trajectories:
                traj = np.array(traj)
                plt.plot(traj[::downsample, 0], traj[::downsample, 1], alpha=0.7)
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Branching Stochastic Process in 2D')
            plt.show()
            
    def downsample(self, downsample_rate=1):
        return self.trajectories[:, ::downsample_rate, :]

    def marginals(self, downsample_rate=1):
        # returns an N_traj x Nt x d array
        # note that N_traj depends on t though
        xs_data = []
        for j in range(0, self.Nt, downsample_rate):
            trajs = np.unique(self.trajectories[:, j, :], axis=0).tolist()
            xs_data.append(trajs)

        return xs_data
    
    def get_marginal_data(self, downsample_rate=1):
        ts_data = self.ts[::downsample_rate]
        xs_data = self.marginals(downsample_rate=downsample_rate)
        return ts_data, xs_data
    
    def mask_non_unique(self, downsample_rate=1):
        N_traj, Nt, d = self.trajectories.shape
        
        for i in range(Nt):
            unique, counts = np.unique(self.trajectories[:, i, :], axis=0, return_counts=True)
            non_unique_values = unique[counts > 1]
            
            for val in non_unique_values:
                mask = np.all(self.trajectories[:, i, :] == val, axis=1)
                idx_to_keep = np.where(mask)[0][0]  # Keep the first occurrence
                mask[idx_to_keep] = False  # Ensure one value remains
                self.trajectories[:, i, :][mask] = np.nan  # Set others to NaN
        
        return self.trajectories[:, ::downsample_rate, :]

    def get_params(self):
        return {
            'A': self.A,
            'G': self.G,
            'dt': self.dt,
            'Nt': self.Nt,
            'T': self.T,
            'N_traj': self.N_traj,
            'branch_times': self.branch_times,
            'growth_rate': self.growth_rate
        }
    
    def save_file(self, path, downsample_rate=1):
        import h5py
        with h5py.File(path, 'w') as f:
            xs_data = self.mask_non_unique(downsample_rate=downsample_rate)
            ts_data = self.ts[::downsample_rate] # self.trajectories[:, ::downsample_rate, :]
            
            # ts_data, xs_data = self.get_marginal_data(downsample_rate=downsample_rate)
            
            dset_xs_data = f.create_dataset('xs_data', data=xs_data)
            dset_ts_data = f.create_dataset('ts_data', data=ts_data)
            
            dset_xs_data.attrs["downsample_rate"] = downsample_rate
            for key, value in self.get_params().items():
                dset_xs_data.attrs[key] = value
                
        print(f"Data saved to {path}")
        
    def load_data(self, path):
        import h5py
        f = h5py.File(path)
        return f