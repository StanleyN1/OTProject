import numpy as np
import matplotlib.pyplot as plt
from generate_data import BranchingStochasticProcess
import os

# Set the output directory (change this to your desired path)
output_dir = "src/data"  # Default directory

# Allow specifying directory through command line argument
import sys
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
    print(f"Using output directory: {output_dir}")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Define parameters that are common across all simulations
dt = 1e-2   # Time step as requested
Nt = 1500  # Total number of time steps as requested
N_initial = 50  # Number of initial particles as requested

# Define diffusion matrix G (same for all simulations)
G = np.array([[0.1, 0], 
              [0, 0.1]])

# Define A matrices with different eigenvalue properties
# 1. A matrix with positive eigenvalues
A_positive = np.array([[0.5, 0.1],
                       [0.1, 0.5]])
# Eigenvalues: 0.6, 0.4 (both positive)

# 2. A matrix with negative eigenvalues
A_negative = np.array([[-0.5, 0.1],
                       [0.1, -0.5]])
# Eigenvalues: -0.4, -0.6 (both negative)

# 3. A matrix with purely imaginary eigenvalues
A_imaginary = np.array([[0, -0.5],
                        [0.5, 0]])
# Eigenvalues: 0.5i, -0.5i (purely imaginary)

# Generate initial positions - same for all simulations
np.random.seed(42)  # For reproducibility
initial_positions = np.random.normal(0, 0.1, (N_initial, 2))

# Define the matrix configurations
matrix_configs = [
    {"name": "positive", "A": A_positive},
    {"name": "negative", "A": A_negative},
    {"name": "imaginary", "A": A_imaginary}
]

# Define noise level 
# noise_configs = [
#     {"name": "001", "noise_level": 0.01},
#     {"name": "01", "noise_level": 0.1},
#     {"name": "1", "noise_level": 1}
# ]


# Define downsample rates
downsample_rates = [10,50]

# Function to generate the data for each configuration
def generate_data(A, matrix_name, downsample_rate):
    # Set up the branching stochastic process
    bsp = BranchingStochasticProcess(A=A, G=G, dt=dt, Nt=Nt)
    
    # Simulate with zero growth rate as requested
    growth_rate = 0.0  # No branching
    bsp.simulate(initial_positions, growth_rate=growth_rate)
    
    # Save the data with the specified downsample rate
    filename = f"{output_dir}/data_{matrix_name}_ds{downsample_rate}.h5"
    bsp.save_file(filename, downsample_rate=downsample_rate)
    
    # Plot and save trajectories
    # plt.figure(figsize=(8, 6))
    # downsampled_trajectories = bsp.downsample(downsample_rate)
    # for traj in downsampled_trajectories:
    #     plt.plot(traj[:, 0], traj[:, 1], alpha=0.3)
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.title(f'Branching Process - {matrix_name.capitalize()} Eigenvalues (DS: {downsample_rate})')
    # plt.savefig(f"{output_dir}/plot_{matrix_name}_ds{downsample_rate}.png")
    # plt.close()
    
    return bsp

# Generate data for all configurations
for config in matrix_configs:
    matrix_name = config["name"]
    A = config["A"]
    print(f"\nGenerating data for {matrix_name} eigenvalues:")
    eigenvalues = np.linalg.eigvals(A)
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    
    for ds_rate in downsample_rates:
        print(f"\nGenerating with downsample rate {ds_rate}...")
        # Generate data
        bsp = generate_data(A, matrix_name, ds_rate)
    
        # Print some stats about the generated data
        print(f"Generated {bsp.N_traj} trajectories")
        print(f"Saved to: {output_dir}/data_{matrix_name}_ds{ds_rate}.h5")

print("\nAll datasets generated successfully!")