from stanley_appex.estimation import *
from stanley_appex.plotting import *
from stanley_appex.utils import *
from stanley_appex.generate_data import *
import numpy as np
import matplotlib.pyplot as plt
import timeit
import wandb

# 1d, 2d, 5d, 10d
# random -1 +1 matrix with certain percentage of sparsity
# for each dimension, generate 100 matrices and run algorithms on each
# violin plot for each dimension

# how many trajectories, how many time steps?
        
def algorithm(ts_data, xs_data, A_guess, H_guess, N_sample, appex_maxiters):
    
    start = timeit.default_timer()
    As, Hs, Pis = appex_rectangle(xs_data, ts_data, A_guess, H_guess, N_sample=N_sample, tol=1e-5, maxiters=appex_maxiters, print_out=1, save_coupling=True)
    end = timeit.default_timer()
    
    algorithm_time = end - start
    
    return As, Hs, Pis, algorithm_time

def run_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # lambda_func = lambda X: 0.0 # 1.0 # 1.0 # Constant branching rate
        process = BranchingStochasticProcess(np.array(config.A), np.array(config.G), dt=config.dt, Nt=config.Nt, N_traj=config.N_init)
        
        X0 = np.random.multivariate_normal(np.zeros(process.d), np.diag(np.ones(process.d)), process.N_traj) # np.random.normal(0, 1, (N_init, A.shape[1]))
        process.simulate(X0, growth_rate=config.growth_rate)
        
        ts_data, xs_data = process.get_marginal_data(config.downsample_rate)
        
        N_sample = config.N_sample_scale*config.N_init
        As, Hs, Pis, algorithm_time = algorithm(ts_data, xs_data, np.array(config.A_guess), np.array(config.H_guess), N_sample, config.appex_maxiters)
        A_errors, H_errors = matrix_errors(process, As, Hs, config.downsample_rate, plot=False)
        
        A_est = As[-1]
        H_est = Hs[-1]
        A_final_error = A_errors[-1]
        H_final_error = H_errors[-1]
        
        wandb.log({"A_est": A_est, "H_est": H_est, "A_errors": A_errors, "H_errors": H_errors, "A_final_error": A_final_error, "H_final_error": H_final_error, "algorithm_time": algorithm_time})

if __name__ == "__main__":
    import wandb
    wandb.login()
    A = [[1.0, -0.5],
         [0.5, 1.0]]
    G = [[0.5, 0.75],
         [0.75, 1.0]]
    A_guess = [[0.0, 0.0],
               [0.0, 0.0]]
    H_guess = [[1.0, 0.0],
               [0.0, 1.0]]
    sweep_config = {
        'method': 'grid',
        "metric": {"goal": "minimize", "name": "A_final_error"},
    }
    parameters_dict = {
        'downsample_rate' : {
            'values' : [100, 250, 500, 1000, 2000]
        },
        'N_init' : {
            'values': [50, 100, 250, 500]
        },
        'N_sample_scale' : {
            'value' : 5
        },
        'appex_maxiters' : {
            'value' : 20
        },
        'dt' : {
            'value' : 0.0001
        },
        'Nt' : {
            'values' : [15000, 30000]
        },
        'A' : {
            'value' : A
        },
        'G' : {
            'value' : G
        },
        'A_guess' : {
            'value' : A_guess
        },
        'H_guess' : {
            'value' : H_guess
        },
    }
    num_runs = np.prod([len(parameters_dict[key]['values']) for key in parameters_dict if 'values' in parameters_dict[key]])

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="appex-parameter-sweep")
    wandb.agent(sweep_id, run_sweep, count=num_runs)