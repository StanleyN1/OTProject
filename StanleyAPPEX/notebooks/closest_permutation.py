import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import norm

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

# # Example usage
# if __name__ == "__main__":
#     # Input matrix
#     A = np.array([[0.8, 0.2, 0.1],
#                   [0.3, 0.7, 0.4],
#                   [0.5, 0.6, 0.9]])

#     # Find closest permutation matrix and compute Frobenius norm
#     P, dist = closest_permutation_matrix(A, norm_type='fro')

#     print("Input Matrix A:")
#     print(A)
#     print("\nClosest Permutation Matrix P:")
#     print(P)
#     print(f"\nFrobenius Norm between A and P: {dist}")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_two_matrices(mat1, mat2, cmap1="Blues", cmap2="Reds", norm_type="fro"):
    """
    Visualizes two matrices in one heatmap with different colormaps.
    
    Parameters:
        mat1 (ndarray): First matrix (should have the same shape as mat2).
        mat2 (ndarray): Second matrix (should have the same shape as mat1).
        cmap1 (str): Colormap for the first matrix.
        cmap2 (str): Colormap for the second matrix.
        norm_type (str): Matrix norm type to compute difference between mat1 and mat2.
        
    Returns:
        None
    """
    # Validate input dimensions
    if mat1.shape != mat2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    # Create masks for the two matrices
    mask1 = np.triu(np.ones_like(mat1, dtype=bool))  # Upper triangle for mat1
    mask2 = np.tril(np.ones_like(mat2, dtype=bool))  # Lower triangle for mat2

    # Create a figure and axis
    plt.figure(figsize=(8, 6))

    # Plot the first matrix (upper triangle)
    sns.heatmap(mat1, mask=~mask1, cmap=cmap1, cbar=False, annot=True, fmt=".2f", linewidths=0.5)

    # Plot the second matrix (lower triangle)
    sns.heatmap(mat2, mask=~mask2, cmap=cmap2, cbar=False, annot=True, fmt=".2f", linewidths=0.5)

    # Add a color bar for each colormap
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap1), label="Matrix 1 Scale", orientation="vertical", shrink=0.8)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap2), label="Matrix 2 Scale", orientation="vertical", shrink=0.8)

    # Compute and display norm difference between matrices
    norm_diff = np.linalg.norm(mat1 - mat2, ord=norm_type)
    plt.title(f"Combined Heatmap of Two Matrices\nNorm Difference: {norm_diff:.4f}", fontsize=14)
    
    plt.show()

# Example usage
# if __name__ == "__main__":
#     # Generate example matrices
#     mat1 = np.random.rand(5, 5)
#     mat2 = np.random.rand(5, 5)

#     # Visualize the two matrices in one heatmap
#     visualize_two_matrices(mat1, mat2)
# if __name__ == "__main__":
#     pass