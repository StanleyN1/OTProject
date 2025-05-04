import numpy as np
import matplotlib.pyplot as plt
from stanley_appex.estimation import *
from stanley_appex.plotting import *

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
    def __init__(self, A=np.zeros(1), G=np.zeros(1), dt=0.01, Nt=15000, T=1.0, N_traj=1):
        self.A = np.array(A)  # d x d drift matrix
        self.G = np.array(G)  # d x d diffusion matrix
        self.H = self.G@self.G.T
        self.d = self.A.shape[0]
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
    
    def simulate(self, X0, growth_rate=0.0):
        if not callable(growth_rate):
            growth_rate_func = lambda x: growth_rate
            self.growth_rate = growth_rate
        else:
            growth_rate_func = growth_rate
            
        particles = X0  # Initial particles
        self.trajectories = [[X] for X in particles]  # Store all trajectories
        
        self.N_init = X0.shape[0]
        
        t = self.dt
        ti = 1
        while t < self.T:
            new_particles = []
            new_trajectories = []
            for i, X in enumerate(particles):
                
                dW = np.random.randn(len(X)) * np.sqrt(self.dt)
                X_new = X + self.A @ X * self.dt + self.G @ dW

                
                # Branching condition (Poisson process)
                if not np.isclose(self.growth_rate, 0.0):
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
                else:
                    new_particles.append(X_new)
                    new_trajectories.append(self.trajectories[i] + [X_new])
            particles = new_particles
            self.trajectories = new_trajectories
            t += self.dt
            ti += 1
        
        self.trajectories = np.array(self.trajectories)
        self.N_traj = self.trajectories.shape[0]
        print(self.trajectories.shape)
        self.time_marginals = self.marginals()
    
    def simulate_birth_death(self, X0, growth_rate=0.0, death_rate=0.0):
        if not callable(growth_rate):
            growth_rate_func = lambda x: growth_rate
            self.growth_rate = growth_rate
        else:
            growth_rate_func = growth_rate
        if not callable(death_rate):
            death_rate_func = lambda x: death_rate
            self.death_rate = death_rate
        else:
            death_rate_func = death_rate
            
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
                if np.random.rand() < death_rate_func(X) * self.dt:
                    new_trajectories.append(self.trajectories[i] + [np.nan])
                    new_trajectories.append(self.trajectories[i] + [np.nan])
                else:
                    new_particles.append(X_new)
                    new_trajectories.append(self.trajectories[i] + [X_new])

            particles = new_particles
            self.trajectories = new_trajectories
            t += self.dt
            ti += 1
        
        # self.trajectories = np.array(self.trajectories)
        # self.N_traj = self.trajectories.shape[0]
        # print(self.trajectories.shape)
        self.time_marginals = self.trajectories
        
    def simulate_track(self, X0, growth_rate=0.0):
        # BROKEN AS OF 3/20/25
        # self.lineage = list(range(len(X0)))
        
        if not callable(growth_rate):
            growth_rate_func = lambda x: growth_rate
            self.growth_rate = growth_rate
        else:
            growth_rate_func = growth_rate
        
        particles = X0  # Initial particles
        parent_map = {i: i for i in range(len(particles))}  # Track parent indices
        
        self.trajectories = [[X] for X in particles]
        
        t = self.dt
        ti = 1
        while t < self.T:
            new_particles = []
            new_trajectories = []
            new_lineage = []
            next_id = len(parent_map)  # New index for particles
            
            for i, X in enumerate(particles):
                dW = np.random.randn(len(X)) * np.sqrt(self.dt)
                X_new = X + self.A @ X * self.dt + self.G @ dW

                # Branching condition (Poisson process)
                if np.random.rand() < growth_rate_func(X) * self.dt:
                    new_particles.append(X_new)  # Keep the original
                    new_particles.append(X_new)  # Create a new branch at the same position
                    self.branch_times.append(ti + 1)

                    # Track lineage
                    # parent_map[next_id] = parent_map[i]  # First branch retains original parent
                    # parent_map[next_id + 1] = parent_map[i]  # New branch also gets same parent
                    parent_map[next_id] = i
                    parent_map[next_id + 1] = i 
                    # Update trajectories
                    new_trajectories.append(self.trajectories[i] + [X_new])
                    new_trajectories.append(self.trajectories[i] + [X_new])
                    
                    # Update lineage list
                    # new_lineage.append(parent_map[next_id])
                    # new_lineage.append(parent_map[next_id + 1])
                    new_lineage.append(i)
                    new_lineage.append(i)
                    
                    next_id += 2  # Increment ID counter
                else:
                    new_particles.append(X_new)
                    new_trajectories.append(self.trajectories[i] + [X_new])
                    
                    # Track lineage (no branching)
                    
                    # parent_map[next_id] = parent_map[i]
                    # new_lineage.append(parent_map[next_id])
                    
                    parent_map[next_id] = i
                    new_lineage.append(i)
                    
                    next_id += 1

            particles = new_particles
            self.trajectories = new_trajectories
            self.lineage = new_lineage  # Update lineage tracking
            
            t += self.dt
            ti += 1
        self.trajectories = np.array(self.trajectories)
        self.N_traj = self.trajectories.shape[0]
    
    def plot_trajectories(self, downsample=1, dim=2, legend=False):
        plt.figure(figsize=(4, 4))
        if dim == 0:
            for i, traj in enumerate(self.trajectories):
                traj = np.array(traj)
                plt.plot(traj[::downsample, 0], alpha=0.7, label=f'Traj {i}')
            plt.xlabel('Time')
            plt.ylabel('X')
            plt.title('Branching Stochastic Process in 1D')
            
        if dim == 1:
            for i, traj in enumerate(self.trajectories):
                traj = np.array(traj)
                plt.plot(traj[::downsample, 1], alpha=0.7, label=f'Traj {i}')
            plt.xlabel('Time')
            plt.ylabel('X')
            plt.title('Branching Stochastic Process in 1D')
            
        if dim == 2:
            for i, traj in enumerate(self.trajectories):
                traj = np.array(traj)
                plt.plot(traj[::downsample, 0], traj[::downsample, 1], alpha=0.7, label=f'Traj {i}')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Branching Stochastic Process in 2D')

        if legend:
            plt.legend()
        plt.show()
        
            
    def downsample(self, downsample_rate=1):
        return self.trajectories[:, ::downsample_rate, :]
    
    def get_branch_times(self, downsample_rate=1):
        branch_times_data = np.array(self.branch_times) // downsample_rate
        return branch_times_data

    def marginals(self, downsample_rate=1):
        # returns an N_traj x Nt x d array
        # note that N_traj depends on t though
        xs_data = []
        for j in range(0, self.Nt - 1, downsample_rate):
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
    
    def mask_non_unique_backward(self, downsample_rate=1):
        N_traj, Nt, d = self.trajectories.shape
        
        for i in range(1, Nt):
            unique, counts = np.unique(self.trajectories[:, i, :], axis=0, return_counts=True)
            non_unique_values = unique[counts > 1]
            
            for val in non_unique_values:
                mask = np.all(self.trajectories[:, i, :] == val, axis=1)
                idx_to_keep = np.where(mask)[0][0]  # Keep the first occurrence
                mask[idx_to_keep] = False  # Ensure one value remains
                self.trajectories[:, i-1, :][mask] = np.nan  # Set others to NaN
        
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
            
            f.create_dataset('trajectories', data=self.trajectories)
            dset_xs_data = f.create_dataset('xs_data', data=xs_data)
            dset_ts_data = f.create_dataset('ts_data', data=ts_data)
            
            dset_xs_data.attrs["downsample_rate"] = downsample_rate
            for key, value in self.get_params().items():
                dset_xs_data.attrs[key] = value
                
        print(f"Data saved to {path}")
    
    def run_appex(self, ts_data=None, xs_data=None, A_guess=None, H_guess=None, downsample_rate=1, N_sample=100, maxiters=5, print_out=10, save_coupling=True, reverse=False):
        if self.ts_data is not None and self.xs_data is not None:
            xs_data = self.xs_data.transpose((1, 0, 2))[::downsample_rate]
            ts_data = self.ts_data[::downsample_rate]
            assert xs_data.shape[0] == len(ts_data), f"Length of xs_data ({len(xs_data)}) and ts_data ({len(self.ts_data)}) must match"
        else:
            if ts_data is None:
                ts_data = self.ts[::downsample_rate]
            if xs_data is None:
                xs_data = self.marginals(downsample_rate=downsample_rate)

        assert len(ts_data) == len(xs_data), f"Length of ts_data ({len(ts_data)}) and xs_data ({len(xs_data)}) must match"
        assert len(xs_data) > 0, "xs_data must not be empty"
        assert len(ts_data) > 0, "ts_data must not be empty"
        assert len(xs_data[0][0]) == self.d, "xs_data must have the same dimension as the process"
        
        print("N traj", self.N_traj, "N time steps", ts_data.shape[0])

        ridge_lambda = 0.0

        if A_guess is None:
            A_guess = np.zeros((self.d, self.d))
        if H_guess is None:
            H_guess = np.eye(self.d)
        
        As, Hs, Pis = appex_rectangle(xs_data, ts_data, A_guess, H_guess, N_sample=N_sample, ridge_lambda=ridge_lambda, tol=1e-5, maxiters=maxiters, print_out=print_out, save_coupling=save_coupling, reverse=reverse)

        A_est = As[-1]
        H_est = Hs[-1]
        Pi_est = Pis[-1]
        
        self.A_est = A_est
        self.H_est = H_est
        self.Pi_est = Pi_est
        self.xs_data = xs_data
        self.ts_data = ts_data
        self.A_error, self.H_error = estimation_error(As, self.A, Hs, self.H)
        
    def plot_sampling(self, A_est=None, H_est=None, Pi_est=None, skip=1, downsample_rate=1, N_sample=100, plot_dim=2, reverse=True):
        if self.xs_data is None or self.ts_data is None:
            raise ValueError("xs_data and ts_data must be set before plotting by running `run_appex`.")
        
        plot_sampling(self.xs_data[skip:], self.ts_data, self.Pi_est[skip:], self.downsample(downsample_rate), self.get_branch_times(downsample_rate), N_sample=N_sample, plot_dim=plot_dim, reverse=reverse)
    
    def load_data(self, path):
        import h5py
        f = h5py.File(path)

        self.xs_data = f['xs_data'][:]
        self.ts_data = f['ts_data'][:]
        self.A = f['xs_data'].attrs['A']
        self.G = f['xs_data'].attrs['G']
        self.dt = f['xs_data'].attrs['dt']
        self.Nt = f['xs_data'].attrs['Nt'] + 1
        self.T = f['xs_data'].attrs['T']
        self.N_traj = f['xs_data'].attrs['N_traj']
        self.branch_times = f['xs_data'].attrs['branch_times']
        self.growth_rate = f['xs_data'].attrs['growth_rate']
        self.d = self.A.shape[0]
        self.H = self.G @ self.G.T
        self.trajectories = f['trajectories'][:]
        
        return f