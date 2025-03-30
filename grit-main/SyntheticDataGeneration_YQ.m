function [scdata, Tgrid] = SyntheticDataGeneration_YQ(A_true, b_true, x0, n_timepoints, n_cells, epsilon,output_dir)
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
    % scdata : cell array of matrices
    %     Generated gene expression matrices at different time points
    
    n_genes = size(A_true, 1);
    scdata = cell(1, n_timepoints);

    Tgrid = 1:n_timepoints;
    
    % Generate initial time point
    x0_samples = mvnrnd(x0, eye(n_genes), n_cells)';
    scdata{1} = x0_samples;
    
    % Generate subsequent time points
    for t = 2:n_timepoints
        prev_timepoint = scdata{t-1};
        
        % Propagate cells
        noise = sqrt(epsilon) * randn(n_genes, n_cells);
        next_timepoint = (eye(n_genes) + A_true) * prev_timepoint + repmat(b_true(:), 1, n_cells) + noise;
        
        scdata{t} = next_timepoint;
    end
    % Save data to CSV files if output_dir is provided
    if nargin >= 7 && ~isempty(output_dir)
        % Create directory if it doesn't exist
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % Save metadata
        metadata.n_genes = n_genes;
        metadata.n_timepoints = n_timepoints;
        metadata.n_cells = n_cells;
        metadata.epsilon = epsilon;
        save(fullfile(output_dir, 'metadata.mat'), 'metadata', 'A_true', 'b_true', 'x0', 'Tgrid');
        
        % Save time points as CSV files
        for t = 1:n_timepoints
            % Transpose to get cells as rows, genes as columns
            data_matrix = scdata{t}';
            
            % Create column headers (Gene1, Gene2, ...)
            headers = cell(1, n_genes);
            for g = 1:n_genes
                headers{g} = sprintf('Gene%d', g);
            end
            
            % Create table with headers
            data_table = array2table(data_matrix, 'VariableNames', headers);
            
            % Save to CSV
            filename = fullfile(output_dir, sprintf('timepoint_%03d.csv', t));
            writetable(data_table, filename);
            
            fprintf('Saved timepoint %d to %s\n', t, filename);
        end
        
        fprintf('Data successfully saved to %s\n', output_dir);
    end
end
