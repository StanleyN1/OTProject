from GRIT_src import *
import numpy as np


def run_sweep():
    # Initialize wandb
    wandb.init(project="grit-parameter-sweep")
    
    # Get parameters from wandb config
    config = wandb.config
    
    # Track metrics across all A matrices
    all_a_errors = []
    all_b_errors = []
    
    # Run for multiple A matrices
    for i in range(config.n_matrices):
        # Generate sparse A matrix (40% zeros, rest +1/-1)
        A_true = generate_sparse_A(config.n_genes, sparsity=0.6)
        
        # Generate data
        np.random.seed(i)
        b_true = np.zeros(config.n_genes)
        x0_mean = np.random.random(config.n_genes)
        scdata,Tgrid = generate_discrete_time_system_data(
        A_true =  A_true, 
        b_true = b_true, 
        x0 = x0_mean, 
        n_timepoints = config.n_timepoints, 
        dt= config.dt, 
        n_cells=config.n_cells, 
        epsilon= config.epsilon
    )
        
        # Run GRIT
        pred_A = GRIT_MATLAB_Reduced(scdata, Tgrid, epsilon=config.epsilon)
        
        # Calculate error between true and learned A matrices
        A_error = np.sum((pred_A[:,:-1]- A_true) ** 2)
        b_error = np.sum((pred_A[:,-1]- b_true) ** 2)
        #total_error = A_error + b_error

        # Store metrics
        all_a_errors.append(A_error)
        all_b_errors.append(b_error)
    
    # Log metrics
    wandb.log({
        "avg_A_matrix error": np.mean(all_a_errors),
        "std_A_matrix error": np.std(all_a_errors),
        "avg_b_vector error": np.mean(all_b_errors),
        "std_b_vector error": np.std(all_b_errors)
    })

if __name__ == "__main__":
    import wandb
    wandb.login()
    sweep_config={
        'method': 'grid',
        'parameters': {
            'n_matrices': {
                'values': [100]
            },
            'dt': {
                'values': [0.001, 0.01, 0.1]
            },
            'n_cells': {
                'values': [10, 100]
            },
            'n_timepoints': {
                'values': [10, 30, 50]
            },
            'n_genes': {
                'values': [1, 2, 5, 10]
            },
            'epsilon': {
                'values': [0.01, 0.1, 1]
            }
        }
    }
    parameters_dict = sweep_config['parameters']
    num_runs = np.prod([len(parameters_dict[key]['values']) for key in parameters_dict if 'values' in parameters_dict[key]])
    sweep_id = wandb.sweep(sweep_config, project="GRIT-parameter-sweep")
    wandb.agent(sweep_id, run_sweep)
    
