from stanley_appex.estimation import *
from stanley_appex.plotting import *
from stanley_appex.utils import *
from stanley_appex.generate_data import *
import numpy as np
import matplotlib.pyplot as plt
import timeit
import wandb
import h5py

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

def OTsolver_MATLABCODE(mu0, mu1, C, epsilon, uInit=None):
        """
        Optimal Transport solver using Sinkhorn algorithm with entropic regularization.

        Parameters:
        -----------
        mu0 : array_like
            Source distribution (will be reshaped to column vector)
        mu1 : array_like
            Target distribution (will be reshaped to column vector)
        C : array_like
            Cost matrix
        epsilon : float
            Regularization parameter
        uInit : array_like, optional
            Initial value for dual variable u1

        Returns:
        --------
        transport_cost : float
            The optimal transport cost
        reg_cost : float
            The regularization cost
        M : ndarray
            The optimal transport plan (matrix)
        iteration_count : int
            Number of iterations needed
        u1 : ndarray
            Final dual variable
        """
        # Reshape distributions to column vectors
        mu0 = np.reshape(mu0, (-1, 1))
        mu1 = np.reshape(mu1, (-1, 1))
        

        # Check dimensions
        if abs(mu0.shape[0] - C.shape[0]) + abs(mu1.shape[0] - C.shape[1]) >= 1:
            print('dimension error')
            print(f"mu0 dimension: {mu0.shape}")
            print(f"mu1 dimension: {mu1.shape}")
            print(f"C dimension: {C.shape}")
            return None, None, None, None, None

        # Compute Gibbs kernel
        K = np.exp(-C / epsilon)

        # Initialize dual variables
        if uInit is not None:
            u1 = np.reshape(uInit, (-1, 1))
        else:
            u1 = mu1.copy()

        u0 = mu0.copy()
        u0_old = np.ones_like(u0) * 100  # arbitrary large value
        iteration_count = 0

        # Sinkhorn's algorithm main loop
        while np.linalg.norm(np.log(u0_old + 1e-16) - np.log(u0 + 1e-16)) > 1e-3 and iteration_count < 10000:
            u0_old = u0.copy()
            u0 = mu0 / (K @ u1)  # Element-wise division
            u1 = mu1 / (K.T @ u0)  # Element-wise division, K.T is transpose
            iteration_count += 1

        # Compute optimal transport plan
        # Create diagonal matrices from vectors
        diag_u0 = np.diag(u0.flatten())
        diag_u1 = np.diag(u1.flatten())
        M = diag_u0 @ K @ diag_u1

        # Calculate transport cost
        transport_cost = np.sum(M * C)  # Element-wise product

        # Calculate regularization cost
        M_flat = M.flatten()
        # Add small constant to avoid log(0)
        log_M = np.log(M_flat + 1e-16)
        EE = M_flat * log_M

        # Filter out NaN values (which might occur from log(0))
        valid_indices = ~np.isnan(EE)
        reg_cost = np.sum(EE[valid_indices])

        return transport_cost, reg_cost, M, iteration_count, u1

def GRIT(scdata,Tgrid,config,A):
    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference

    Parameters:
    -----------
    scdata : list of numpy arrays
        size: num_timepoints x num_genes x num_cells
        Single-cell data for each time point
    Tgrid : list or numpy array
        Time points for the data
    TFflag : numpy array
        Flags for transcription factors
    branchId : list of numpy arrays
        Branch identifiers for each cell at each time point
    opts : dict, optional
        Options for the algorithm

    Returns:
    --------
    Multiple variables including the gene regulatory matrix A and the transport maps
    """
    branchId = []
    opts=None

    # Initialize output structure
    out = {}

    # Set default options
    if opts is None:
        opts = {}

    if 'epsilon' not in opts:
        opts['epsilon'] = config["noise_strength"] # 0.05
    if 'iterations' not in opts:
        opts['iterations'] = 30
    if 'zeroWeight' not in opts:
        opts['zeroWeight'] = 1
    if 'disp' not in opts:
        opts['disp'] = 'basic'
    if 'par' not in opts:
        opts['par'] = 0
    if 'Nred' not in opts:
        print(len(A))
        opts['Nred'] = len(A) # min(100, round(0.9 * scdata[0].shape[0])+1)
    # if 'maxReg' not in opts:
    #     opts['maxReg'] = 40
    # if 'branchWeight' not in opts:
    #     opts['branchWeight'] = 2
    # if 'signed' not in opts:
    #     opts['signed'] = False

    # Initialize branch IDs if empty
    if not branchId:
        branchId = []
        for jt in range(len(scdata)):
            branchId.append(np.ones((1, scdata[jt].shape[1])))

    # Get dimensions
    ndim = scdata[0].shape[0]  # Number of genes
    # nbr = branchId[0].shape[0]  # Number of branches
    nbr = 1

    # Count cells in samples
    ncell = np.zeros(len(scdata), dtype=int)
    nzeros = 0
    nelements = 0

    # Process each time point
    for jt in range(len(scdata)):
        ncell[jt] = scdata[jt].shape[1]
        nzeros += np.sum(scdata[jt] == 0)
        nelements += scdata[jt].size

        # Check for negative values
        if np.min(scdata[jt]) < 0 and opts['zeroWeight'] < 1:
            opts['zeroWeight'] = 1
            warnings.warn("Negative values in the gene expression data. Zero inflation will not be accounted for.")

    # Process time grid
    Ttot = 0
    if isinstance(Tgrid, list) and all(isinstance(item, (list, np.ndarray)) for item in Tgrid):  # Multiple experiments
        Tg = Tgrid
        Tgrid = []
        indtr = []
        iaux = 0
        indw = []
        indexp = []

        for jex in range(len(Tg)):
            Tgrid.extend(Tg[jex])
            indtr.extend([(i + iaux) for i in range(len(Tg[jex])-1)])

            indw.extend([sum(ncell[:iaux]) + i for i in range(sum(ncell[iaux:iaux+len(Tg[jex])-1]))])

            indexp.extend([jex] * sum(ncell[iaux+1:iaux+len(Tg[jex])]))

            iaux += len(Tg[jex])
            Ttot += max(Tg[jex]) - min(Tg[jex])

        Tgrid = np.array(Tgrid)
        indtr = np.array(indtr)
        indw = np.array(indw)
        indexp = np.array(indexp)
    else:  # Single experiment
        Tgrid = np.array(Tgrid)
        indtr = np.arange(len(scdata)-1) # number of transitions
        indw = np.arange(sum(ncell[:-1])) # number of cells
        Ttot = max(Tgrid) - min(Tgrid)
        indexp = np.ones(sum(ncell[:-1]), dtype=int)

    # Check consistency
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # Calculate variances
    vvs = np.zeros((ndim, len(indtr)))
    for jt in range(len(indtr)):
        for jbr in range(nbr):
            branch_cells = np.where(branchId[indtr[jt]+1][jbr, :] > 0)[0]
            if len(branch_cells) > 0:
                mtemp = np.mean(scdata[indtr[jt]+1][:, branch_cells], axis=1, keepdims=True)
                mtemp[np.isnan(mtemp)] = 0
                vvs[:, jt] += np.sum((scdata[indtr[jt]+1][:, branch_cells] - mtemp)**2, axis=1)

        vvs[:, jt] = vvs[:, jt] / ncell[indtr[jt]+1]
    # Corrected code - explicitly handle the broadcasting
    weighted_vvs = np.zeros_like(vvs)
    for jt in range(len(indtr)):
        weighted_vvs[:, jt] = ncell[indtr[jt]] * vvs[:, jt]

    vvs = 0.5 + 0.2 * np.sum(weighted_vvs, axis=1, keepdims=True) / np.sum(ncell[indtr]) + 0.8 * vvs
    vvs = vvs**(-0.5)

    # Build regression matrices
    XX = np.empty((ndim + nbr, 0))
    #XX = np.empty((ndim, 0)) #change 1
    DT = np.empty(0)

    for jt in range(len(indtr)):
        data_with_branch = np.vstack([scdata[indtr[jt]], branchId[indtr[jt]]]) #change 2
        XX = np.hstack([XX, data_with_branch])
        #data_at_jt = scdata[indtr[jt]]
        #XX = np.hstack([XX, data_at_jt])

        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])

    YY = np.zeros((ndim, XX.shape[1]))

    # Regularization weights
    D = 0.01 * Ttot * np.diag(XX @ XX.T) / XX.shape[1]
    D[-nbr:] = 10 * D[-nbr:] #change 3

    # Reduced weights for zeros
    WW = np.ones((ndim, sum(ncell)))
    for jt in range(len(scdata)):
        indices = slice(sum(ncell[:jt]), sum(ncell[:jt+1]))
        WW[:, indices] = 1 - (1 - opts['zeroWeight']) * (scdata[jt] < 1e-10)

    # Calculate correlations and dimension reduction
    Xc = XX[:ndim, :]
    indmiss = np.setdiff1d(np.arange(len(scdata)), indtr)

    for im in indmiss:
        Xc = np.hstack([Xc, scdata[im]])

    Xc = Xc - np.mean(Xc, axis=1, keepdims=True)


    nred = min(opts['Nred'], min(Xc.shape))

    # If matrix is large, use randomized SVD approach for efficiency
    if max(Xc.shape) > 10000:
        # Randomized SVD - efficient for large matrices and when we only need top k components
        from sklearn.utils.extmath import randomized_svd
        Ured, S, _ = randomized_svd(Xc, n_components=nred, random_state=42)
    else:
        # For smaller matrices, use numpy's full SVD and take top components
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        Ured = U[:, :nred]
        S = S[:nred]

    corNet = Xc @ Xc.T
    sc = np.diag(corNet)**(-0.5)
    corNet = sc.reshape(-1, 1) * corNet * sc
    out['vars'] = (S**2) / np.sum(Xc**2)
    indmiss = np.hstack([[0], indmiss])

    # Scale XX by time differences
    XX = DT * XX

    # Initialize iterative solution
    A = np.zeros((ndim, ndim+nbr)) # change 4
    #A = np.ones((ndim, ndim))
    difs = np.zeros(opts['iterations'])
    J = np.zeros(opts['iterations'])
    out['its'] = np.zeros((opts['iterations'], len(indtr)))

    rat = [1] * len(indtr)
    m_rat = 1
    uFin = [1] * len(indtr)
    transportMap = [None] * len(scdata)

    # Prepare for main iteration loop
    convergenceProblemIndicator = [0] * len(indtr)
    tMap = [None] * len(indtr)
    Jadd = [0] * len(indtr)
    its = [0] * len(indtr)
    der = [None] * len(indtr)

    # Main iterative loop
    for jiter in range(opts['iterations']):
        kreg = min(0.5 + 0.7 * (jiter+1) / opts['iterations'], 1)
        Aold = A.copy()

        # Process each transition
        for jt in range(len(indtr)):
            # Propagated and target points
            # change 5
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ np.vstack([scdata[indtr[jt]], branchId[indtr[jt]]])
            #X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]
            X1 = scdata[indtr[jt]+1]

            # Cost matrix
            ett0 = np.ones((X0.shape[1], 1))
            ett1 = np.ones((X1.shape[1], 1))


            vX0 = vvs[:, jt].reshape(-1, 1) * X0
            vX1 = vvs[:, jt].reshape(-1, 1) * X1

            UvX0 = Ured.T @ vX0
            UvX1 = Ured.T @ vX1

            sum_UvX0_squared = np.sum(UvX0**2, axis=0).reshape(-1, 1)
            sum_UvX1_squared = np.sum(UvX1**2, axis=0)


            C = sum_UvX0_squared @ ett1.T - 2 * UvX0.T @ UvX1 + ett0 @ sum_UvX1_squared.reshape(1, -1)


            # only for same amount of cell in each time point
            mu0 = np.ones(np.shape(X0)[1])/np.shape(X0)[1]
            mu1 = np.ones(np.shape(X1)[1])/np.shape(X1)[1]


            if jiter == 0:
                uInit = mu1
            else:
                uInit = uFin[jt]

            # Solve optimal transport problem
            epsloc = opts['epsilon']
            failInd = True
            failedSinkhornIterations = -1

            M = None
            uFinal = None
            transport_cost = 0
            reg_cost = 0
            iteration_count = 0

            while failInd and failedSinkhornIterations < 10:
                transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, opts['epsilon'], uInit)
                # transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, epsloc * np.median(C), uInit)
                #transport_plan = ot.sinkhorn(a, b, cost_matrix, epsilon, method=method)

                failInd = np.sum(np.isnan(M)) > 0
                epsloc = 1.5 * epsloc
                failedSinkhornIterations += 1

            if jiter == opts['iterations'] - 1 and failedSinkhornIterations > 0:
                convergenceProblemIndicator[jt] = 1

            tMap[jt] = M
            uFin[jt] = uFinal
            M = M / np.sum(M, axis=1, keepdims=True)
            Jadd[jt] = transport_cost + opts['epsilon'] * np.median(C) * reg_cost
            its[jt] = iteration_count

            # Estimate derivatives
            ww_slice = WW[:, sum(ncell[:indtr[jt]]):sum(ncell[:indtr[jt]+1])]

            M_product = ww_slice @ M
            X1M = X1 @ M.T
            X0M = X0 @ M

            der[jt] = ((X1M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5
            # der[jt] = ((X0M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

        # Collect derivatives
        iaux = 0
        for jt in range(len(indtr)):
            YY[:, iaux:iaux + der[jt].shape[1]] = der[jt]
            J[jiter] += Jadd[jt]
            iaux += der[jt].shape[1]
            transportMap[indtr[jt]] = tMap[jt]
            out['its'][jiter, jt] = its[jt]

        # Solve for A using regression
        Anew = np.zeros_like(A)

        for jg in range(ndim):
            # Set of regressors are the TFs and the target gene itself
            # TFloc = TFflag.copy().flatten()
            # TFloc[jg] = True

            WV = np.zeros(YY.shape[1])
            iaux = 0
            for jt in range(len(indtr)):
                WV[iaux:iaux + ncell[indtr[jt]]] = vvs[jg, jt]**2
                iaux += ncell[indtr[jt]]

            # Weight matrices
            W_diag = WV * WW[jg, indw]
            #XX_TF = XX[TFloc, :]

            # Weighted regression with regularization
            #weighted_XX = XX_TF * W_diag
            weighted_XX = XX * W_diag
            weighted_YY = YY[jg, :] * W_diag

            #reg_matrix = weighted_XX @ XX_TF.T + np.diag(D[TFloc])
            reg_matrix = weighted_XX @ XX.T + np.diag(D)
            #Anew[jg, TFloc] = weighted_YY @ XX_TF.T @ np.linalg.inv(reg_matrix)
            Anew[jg, :] = weighted_YY @ XX.T @ np.linalg.inv(reg_matrix)



        # Regularized update
        A = (1 - kreg) * Aold + kreg * Anew

        # Check progress
        difs[jiter] = np.sqrt(np.sum((A - Aold)**2))

        if opts['disp'] == 'all':
            print(f"Iteration {jiter+1}/{opts['iterations']} done.")

    # Final A is the unregularized one
    A = Anew

    # Check for and report about convergence problems
    expnr = 0
    for jt in range(len(convergenceProblemIndicator)):
        if convergenceProblemIndicator[jt] == 1:
            if expnr == 0:
                warnings.warn("Convergence problems detected and regularisation increased")
                print("Check the following matrices for outliers:")

            expnr = 1 + indtr[jt] - jt
            tpnr = indtr[jt] - indmiss[expnr]
            print(f"* Experiment {expnr}, time points {tpnr} and {tpnr+1}")

#     if opts['disp'] != 'off':
#         print("Model identification complete.")

    #OUTPUT = [XX, YY, transportMap, J, A, D, WW, corNet, indw, indexp, TFflag, difs, out, opts]
    return A[:, :-1], transportMap

h5py.File
def run_comparison(A, config=None):
        
        """START Simulate data"""
        # lambda_func = lambda X: 0.0 # 1.0 # 1.0 # Constant branching rate
        
        G = config["noise_strength"]*np.eye(len(A))
        process = BranchingStochasticProcess(np.array(A), np.array(G), dt=config["dt"], Nt=config["Nt"], N_traj=config["N_init"])
        X0 = np.random.multivariate_normal(np.zeros(process.d), np.diag(np.ones(process.d)), process.N_traj) # np.random.normal(0, 1, (N_init, A.shape[1]))
        process.simulate(X0, growth_rate=0.0)
        # print(process.trajectories.shape)
        
        ts_data, xs_data = process.get_marginal_data(config["downsample_rate"])
        """END Simulate data"""
        process.plot_trajectories()
        """START APPEX"""
        N_sample = config["N_sample_scale"]*config["N_init"]
        A_guess_appex = np.zeros_like(np.array(A))
        H_guess_appex = G@G.T
        A_s, Hs, Pis, algorithm_time = algorithm(ts_data, xs_data, np.array(A_guess_appex), np.array(H_guess_appex), N_sample, config["appex_maxiters"])
        A_errors, H_errors = matrix_errors(process, A_s, Hs, config["downsample_rate"], plot=False)
        
        A_est = A_s[-1]
        H_est = Hs[-1]
        A_final_error = A_errors[-1]
        H_final_error = H_errors[-1]
        """END APPEX"""
        
        """START GRIT"""
        grit_data = process.downsample(config["downsample_rate"]).transpose((1, 2, 0))
        A_grit, transport_map = GRIT(grit_data, ts_data,config,np.array(A))
        A_grit_error = np.linalg.norm(A - A_grit, ord="fro")
        """END GRIT"""
        
        return ({"A_appex": A_est, "H_appex": H_est, "A_grit": A_grit, "A_errors": A_errors, "H_errors": H_errors, "A_appex_final_error": A_final_error, "H_appex_final_error": H_final_error, "A_grit_final_error": A_grit_error, "A": A, "transport_map": transport_map})

def generate_A(d, scale=0.5, n_matrices=50, p_zero=0.7):
    p_other = (1-p_zero)/2
    A = 0.5*np.random.choice([-1, 0, 1], p=[p_other, p_zero, p_other],size=(n_matrices, d, d))
    return [A[i, :, :].tolist() for i in range(A.shape[0])]

As = generate_A(d=2, n_matrices=25, p_zero=0.1) + generate_A(d=5, n_matrices=50) + generate_A(d=10, n_matrices=100) + generate_A(d=20, n_matrices=100)

if __name__ == "__main__":

    config_all = {
        'downsample_rate' : {
            'values' : [100, 500, 1000, 2000]
        },
        'N_init' : {
            'values': [50, 100, 250, 500]
        },
        'noise_strength': {
            'values': [0.05, 0.1, 0.5, 1.0, 2.0]
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
            'value' : 1000 # 15000
        },
    }
    
    def select_config(config_all):
        selected_config = {}
        for key, value in config_all.items():
            if 'values' in value:
                # selected_config[key] = np.random.choice(value['values'])
                selected_config[key] = value['values'][0]
            elif 'value' in value:
                selected_config[key] = value['value']
        return selected_config

    config = select_config(config_all)
    print(config)
    
    np.random.seed(0)
    A = generate_A(d=10, n_matrices=1, p_zero=0.8)[0]
    output = run_comparison(A, config)
    print("A", output["A"])
    print("A_appex", output["A_appex"])
    print("A_grit", output["A_grit"])
    print("APPEX ERROR", output["A_appex_final_error"])
    print("GRIT ERROR", output["A_grit_final_error"])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(output["A"])
    plt.colorbar()
    plt.title("True A")
    plt.subplot(1, 3, 2)
    plt.imshow(output["A_grit"])
    plt.colorbar()
    plt.title("GRIT")
    plt.subplot(1, 3, 3)
    plt.imshow(output["A_appex"])
    plt.colorbar()
    plt.title("APPEX")
    
    plt.suptitle("GRIT ERROR: {:.2f}, APPEX ERROR: {:.2f}".format(output["A_grit_final_error"], output["A_appex_final_error"]))
    plt.tight_layout()
    
    plt.show()
    
    # print(output['transport_map'][:-1])
    # plt.imshow(output['transport_map'])
    # plt.show()
    
    
    