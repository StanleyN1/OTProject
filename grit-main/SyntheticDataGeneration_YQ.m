function [Y_list, T_grid] = SyntheticDataGeneration(A_true, b_true, x0, n_timepoints, n_cells, epsilon)
    % Generate synthetic data based on the discrete-time system equation
    %
    % Parameters:
    % -----------
    % A_true : matrix
    %     True system matrix
    % b_true : vector
    %     True constant load vector
    % x0 : vector
    %     Initial distribution mean
    % n_timepoints : int
    %     Number of time points to generate
    % n_cells : int
    %     Number of cells per time point
    % epsilon : float
    %     Noise intensity
    %
    % Returns:
    % --------
    % Y_list : cell array of matrices
    %     Generated gene expression matrices at different time points
    
    n_genes = size(A_true, 1);
    Y_list = cell(1, n_timepoints);

    T_grid = 1:n_timepoints;
    
    % Generate initial time point
    x0_samples = mvnrnd(x0', eye(n_genes), n_cells)';
    Y_list{1} = x0_samples;
    
    % Generate subsequent time points
    for t = 2:n_timepoints
        prev_timepoint = Y_list{t-1};
        
        % Propagate cells
        noise = sqrt(epsilon) * randn(n_genes, n_cells);
        next_timepoint = (eye(n_genes) + A_true) * prev_timepoint + b_true(:) + noise;
        
        Y_list{t} = next_timepoint;
    end
end
