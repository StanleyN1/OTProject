% This script generates synthetic data using SyntheticDataGeneration_YQ 
% and attempts to recover the dynamics using GRITmodelSelect

% Set random seed for reproducibility
rng(42);

% Define system parameters
n_genes = 5;
n_timepoints = 10;
n_cells = 100;
epsilon = 0.1;
% Create true system dynamics

% % Test Case 1
% A_true = -0.1*eye(n_genes);
% b_true = 0.2*ones(n_genes, 1);

%Test Case 2
A_true = 0.1*([
    -0.5, 0.2, 0, 0, 0;
    0.1, -0.1, 0.3, -0.8, 0;
    0, 0.1, -0.7, 0.2, 0;
    0, 0.8, 0.2, -0.8, 0.1;
    0, 0, 0, 0.1, -0.9]);
b_true = 0.3*ones(n_genes);

% %Test Case 3
% A_true = zeros(n_genes, n_genes);
% 
% % Set the diagonal elements to 0.1
% for i = 1:5
%     A_true(i, i) = 0.1;
% end
% 
% % Set the subdiagonal elements to 1
% for i = 2:5
%     A_true(i, i-1) = 1;
% end
% b_true = 0.3*ones(n_genes);





% Random initial state
x0 = rand(n_genes, 1)*5;

% Generate synthetic data
[scdata, Tgrid] = SyntheticDataGeneration_YQ(A_true, b_true, x0, n_timepoints, n_cells, epsilon);

% Define empty branch ID (assuming one branch)
branchId = [];

% Set options for GRITmodelSelect
opts = struct();
%opts.epsilon = 0.1;        % Regularization parameter
opts.iterations = 30;      % Number of iterations
opts.zeroWeight = 1;       % Weight for zero values (1 means no special treatment)
opts.disp = 'basic';       % Display level
opts.par = 0;              % Parallelization off
opts.Nred = min(100, round(0.9*n_genes)); % Dimension reduction
opts.maxReg = 40;          % Maximum regularization
opts.branchWeight = 2;     % Branch weight
opts.signed = false;       % Signed flag

% No transcription factors (empty TFflag means all genes are considered as potential regulators)
TFflag = [];

% Run GRITmodelSelect
[XX, YY, transportMap, J, A_inferred, D, WW, corNet, indw, indexp, TFflag, difs, out, opts] = ...
    GRITmodelSelect(scdata, Tgrid, TFflag, branchId, opts);

% Extract the constant load vector (b) from the last column of A_inferred
b_inferred = A_inferred(:, end);

% Calculate element-wise relative errors
A_element_wise_error = zeros(size(A_true));
for i = 1:n_genes
    for j = 1:n_genes
        if abs(A_true(i,j)) > 1e-10
            A_element_wise_error(i,j) = abs((A_true(i,j) - A_inferred(i,j)) / A_true(i,j));
        else
            A_element_wise_error(i,j) = abs(A_inferred(i,j));
        end
    end
end

b_element_wise_error = zeros(size(b_true));
for i = 1:n_genes
    if abs(b_true(i)) > 1e-10
        b_element_wise_error(i) = abs((b_true(i) - b_inferred(i)) / b_true(i));
    else
        b_element_wise_error(i) = abs(b_inferred(i));
    end
end

% Print element-wise relative errors for A
disp('Element-wise relative errors for A matrix:');
for i = 1:n_genes
    for j = 1:n_genes
        fprintf('A(%d,%d): true=%.4f, inferred=%.4f, rel_error=%.4f\n', ...
            i, j, A_true(i,j), A_inferred(i,j), A_element_wise_error(i,j));
    end
end

% Print element-wise relative errors for b
disp('Element-wise relative errors for b vector:');
for i = 1:n_genes
    fprintf('b(%d): true=%.4f, inferred=%.4f, rel_error=%.4f\n', ...
        i, b_true(i), b_inferred(i), b_element_wise_error(i));
end