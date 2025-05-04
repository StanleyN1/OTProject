import numpy as np
def generate_discrete_time_system_data(A_true, b_true, x0_mean, n_timepoints, dt, n_cells, epsilon, shuffle=False,x0_samples=None):
    
    """
    Generate synthetic data based on the discrete-time system equation
    
    Parameters:
    -----------
    A_true : numpy array
        True system matrix
    b_true : numpy array
        True constant load vector
    x0 : numpy array
        Initial distribution mean
    n_timepoints : int
        Number of time points to generate
    n_cells : int
        Number of cells per time point
    epsilon : float
        Noise intensity
    
    Returns:
    --------
    Y_list : list of numpy arrays
        Generated gene expression matrices at different time points
    """
    A_true = np.array(A_true)
    n_genes = A_true.shape[0]
    Y_list = []
    t_list = []
    
    # Generate initial time point
    if x0_samples is None:
        x0_samples = np.random.multivariate_normal(x0_mean, np.eye(n_genes), n_cells).T
    Y_list.append(x0_samples)
    t_list.append(0)
    
    # Generate subsequent time points
    for _ in range(1, n_timepoints):
        prev_timepoint = Y_list[-1]
        
        # Propagate cells
        noise = np.sqrt(epsilon) * np.random.randn(n_genes, n_cells)
        next_timepoint = prev_timepoint + (A_true @ prev_timepoint + b_true.reshape(-1, 1) + noise) * dt
        
        Y_list.append(next_timepoint)
        t_list.append(t_list[-1]+dt)
        if shuffle:
            np.random.shuffle(Y_list[-1].T) # shuffle cell id

    
    return Y_list, t_list

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

def GRIT_MATLAB_EpsilonInOT(scdata, Tgrid, epsilon):
    TFflag = []
    branchId = []
    opts=None
    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference

    Parameters:
    -----------
    scdata : list of numpy arrays
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
    # Initialize output structure
    out = {}

    # Set default options
    if opts is None:
        opts = {}

    if 'epsilon' not in opts:
        opts['epsilon'] = 0.05
    opts['epsilon'] = epsilon
    if 'iterations' not in opts:
        opts['iterations'] = 30
    if 'zeroWeight' not in opts:
        opts['zeroWeight'] = 1
    if 'disp' not in opts:
        opts['disp'] = 'off'
    if 'par' not in opts:
        opts['par'] = 0
    if 'Nred' not in opts:
        opts['Nred'] = min(100, round(0.9 * scdata[0].shape[0])+1)
    if 'maxReg' not in opts:
        opts['maxReg'] = 40
    if 'branchWeight' not in opts:
        opts['branchWeight'] = 2
    if 'signed' not in opts:
        opts['signed'] = False

    # Initialize parallel processing pool if required
    pool = None
    if opts['par'] > 1:
        pool = multiprocessing.Pool(opts['par'])

    # Initialize branch IDs if empty
    if not branchId:
        branchId = []
        for jt in range(len(scdata)):
            branchId.append(np.ones((1, scdata[jt].shape[1])))

    # Get dimensions
    ndim = scdata[0].shape[0]  # Number of genes
    nbr = branchId[0].shape[0]  # Number of branches

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
        indtr = np.arange(len(scdata)-1)
        indw = np.arange(sum(ncell[:-1]))
        Ttot = max(Tgrid) - min(Tgrid)
        indexp = np.ones(sum(ncell[:-1]), dtype=int)

    # Check consistency
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # Process transcription factor flags
    TFflag = np.array(TFflag).reshape(-1, 1)

    if len(TFflag) < ndim and len(TFflag) > 0:
        warnings.warn("Transcription factor input is interpreted as a list of TF indices")
        TFlist = TFflag.flatten()
        TFflag = np.zeros((ndim, 1), dtype=bool)
        TFflag[TFlist.astype(int)] = True

    if np.sum(TFflag) == 0:
        TFflag = np.ones((ndim, 1), dtype=bool)

    # Include constant load as a "transcription factor"
    TFflag = np.vstack([TFflag, np.ones((nbr, 1), dtype=bool)])

    # Convert to boolean if needed
    if not np.issubdtype(TFflag.dtype, np.bool_):
        TFflag = TFflag > 0

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
    DT = np.empty(0)

    for jt in range(len(indtr)):
        data_with_branch = np.vstack([scdata[indtr[jt]], branchId[indtr[jt]]])
        XX = np.hstack([XX, data_with_branch])

        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])

    YY = np.zeros((ndim, XX.shape[1]))

    # Regularization weights
    D = 0.01 * Ttot * np.diag(XX @ XX.T) / XX.shape[1]
    D[-nbr:] = 10 * D[-nbr:]

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

    # Calculate branch masses
    branchMass = np.zeros((branchId[0].shape[0], 1))
    for jt in range(len(branchId)):
        branchMass += np.sum(branchId[jt], axis=1, keepdims=True)

    branchMass = branchMass / np.sum(branchMass)
    brm = np.zeros((len(branchMass), len(branchId)))

    for jt in range(len(branchId)):
        for jb in range(branchId[jt].shape[0]):
            branch_sum = np.sum(branchId[jt][jb, :]) / np.sum(branchId[jt], axis=1)
            if branch_sum > 0:
                brm[jb, jt] = branchMass[jb, 0] / branch_sum

    if np.max(np.isinf(brm)):
        warnings.warn("Some branches are not present in all time points!")

    brm[np.isinf(brm)] = 1

    # Initialize iterative solution
    A = np.ones((ndim, ndim+nbr))
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
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ np.vstack([scdata[indtr[jt]], branchId[indtr[jt]]])
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

            # Adjust costs for branch transitions
            branch_jumps = (branchId[indtr[jt]].T @ branchId[indtr[jt]+1] == 0)
            C = C * (1 + (opts['branchWeight'] - 1) * branch_jumps)

            # Mass distributions
            br_norm_0 = branchId[indtr[jt]] / np.sum(branchId[indtr[jt]], axis=1, keepdims=True)
            br_norm_1 = branchId[indtr[jt]+1] / np.sum(branchId[indtr[jt]+1], axis=1, keepdims=True)

            mu0 = np.sum(br_norm_0 * brm[:, indtr[jt]].reshape(-1, 1), axis=0)
            mu1 = np.sum(br_norm_1 * brm[:, indtr[jt]+1].reshape(-1, 1), axis=0)

            mu0 = mu0 / np.sum(mu0)
            mu1 = mu1 / np.sum(mu1)

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
                transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, epsloc, uInit)
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

            der[jt] = ((X1M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

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
            TFloc = TFflag.copy().flatten()
            TFloc[jg] = True

            WV = np.zeros(YY.shape[1])
            iaux = 0
            for jt in range(len(indtr)):
                WV[iaux:iaux + ncell[indtr[jt]]] = vvs[jg, jt]**2
                iaux += ncell[indtr[jt]]

            # Weight matrices
            W_diag = WV * WW[jg, indw]
            XX_TF = XX[TFloc, :]

            # Weighted regression with regularization
            weighted_XX = XX_TF * W_diag
            weighted_YY = YY[jg, :] * W_diag

            reg_matrix = weighted_XX @ XX_TF.T + np.diag(D[TFloc])
            Anew[jg, TFloc] = weighted_YY @ XX_TF.T @ np.linalg.inv(reg_matrix)
            

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

    # Clean up pool if it was created
    if pool is not None:
        pool.close()
        pool.join()

    if opts['disp'] != 'off':
        print("Model identification complete.")

    #OUTPUT = [XX, YY, transportMap, J, A, D, WW, corNet, indw, indexp, TFflag, difs, out, opts]
    return A

def GRIT_MATLAB_NO_SVD(scdata,Tgrid,epsilon):

    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference
    WITHOUT branching, zero weight and TF flag
    WITHOUT SVD

    Parameters:
    -----------
    scdata : list of numpy arrays
        Single-cell data for each time point
    Tgrid : list or numpy array
        Time points for the data
    opts : dict, optional
        Options for the algorithm
    Returns:
    --------
    Gene regulatory matrix A and the transport maps
    size of A is (n_genes, n_genes + 1)
    """

    disp = 'off'
    iterations = 30

    # Get dimensions
    ndim = scdata[0].shape[0]  # Number of genes
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

    # Process time grid for Single experiment
    Tgrid = np.array(Tgrid)
    indtr = np.arange(len(scdata)-1)
    indw = np.arange(sum(ncell[:-1]))
    Ttot = max(Tgrid) - min(Tgrid)

    # Check consistency
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # at each time point, calculate variance for each gene over all cells
    vvs = np.zeros((ndim, len(indtr)))
    for jt in range(len(indtr)):
        mtemp = np.mean(scdata[indtr[jt]+1], axis=1, keepdims=True) 
        mtemp[np.isnan(mtemp)] = 0
        vvs[:, jt] += np.sum((scdata[indtr[jt]+1]- mtemp)**2, axis=1)
        vvs[:, jt] = vvs[:, jt] / ncell[indtr[jt]+1] 

    # weight the vvs by number of cells in the previous time step 
    weighted_vvs = np.zeros_like(vvs)
    for jt in range(len(indtr)):
        weighted_vvs[:, jt] = ncell[indtr[jt]] * vvs[:, jt]

    # final variance is 
    vvs = 0.5 + 0.2 * np.sum(weighted_vvs, axis=1, keepdims=True) / np.sum(ncell[indtr]) + 0.8 * vvs
    vvs = vvs**(-0.5)

    # Build regression matrices
    XX = np.empty((ndim + nbr, 0))
    #XX = np.empty((ndim, 0)) #change 1
    DT = np.empty(0)

    for jt in range(len(indtr)):
        branch_padding = np.ones((1, ncell[indtr[jt]])) 
        data_with_branch = np.vstack([scdata[indtr[jt]],branch_padding]) #change 2
        XX = np.hstack([XX, data_with_branch])
        #data_at_jt = scdata[indtr[jt]]
        #XX = np.hstack([XX, data_at_jt])

        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])

    YY = np.zeros((ndim, XX.shape[1]))

    # Regularization weights
    D = 0.01 * Ttot * np.diag(XX @ XX.T) / XX.shape[1]
    D[-nbr:] = 10 * D[-nbr:] #change 3

    #  Weight matrix for each cell, set to 1
    WW = np.ones((ndim, sum(ncell)))

    # Scale XX by time differences
    XX = DT * XX

    A = np.zeros((ndim, ndim+nbr)) # change 4
    #A = np.ones((ndim, ndim))
    difs = np.zeros(iterations)
    J = np.zeros(iterations)

    uFin = [1] * len(indtr)
    transportMap = [None] * len(scdata)

    # Prepare for main iteration loop
    convergenceProblemIndicator = [0] * len(indtr)
    tMap = [None] * len(indtr)
    Jadd = [0] * len(indtr)
    its = [0] * len(indtr)
    der = [None] * len(indtr)

    # Main iterative loop
    for jiter in range(iterations):
        kreg = min(0.5 + 0.7 * (jiter+1) / iterations, 1)
        Aold = A.copy()

        # Process each transition
        for jt in range(len(indtr)):
            # Propagated and target points
            # change 5
            branch_padding = np.ones((1, ncell[indtr[jt]])) 
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ np.vstack([scdata[indtr[jt]], branch_padding])
            #X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]
            X1 = scdata[indtr[jt]+1]

            # Cost matrix
            ett0 = np.ones((X0.shape[1], 1))
            ett1 = np.ones((X1.shape[1], 1))


            vX0 = vvs[:, jt].reshape(-1, 1) * X0
            vX1 = vvs[:, jt].reshape(-1, 1) * X1

            # ignore svd
            #UvX0 = Ured.T @ vX0
            #UvX1 = Ured.T @ vX1

            # ignore svd
            UvX0 = vX0
            UvX1 = vX1

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
            epsloc = epsilon
            failInd = True
            failedSinkhornIterations = -1

            M = None
            uFinal = None
            transport_cost = 0
            reg_cost = 0
            iteration_count = 0

            while failInd and failedSinkhornIterations < 10:
                transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, epsloc * np.median(C), uInit)
                #transport_plan = ot.sinkhorn(a, b, cost_matrix, epsilon, method=method)

                failInd = np.sum(np.isnan(M)) > 0
                epsloc = 1.5 * epsloc
                failedSinkhornIterations += 1

            if jiter == iterations - 1 and failedSinkhornIterations > 0:
                convergenceProblemIndicator[jt] = 1

            tMap[jt] = M
            uFin[jt] = uFinal
            Jadd[jt] = transport_cost + epsilon * np.median(C) * reg_cost
            its[jt] = iteration_count

            M = M / np.sum(M, axis=1, keepdims=True)

            # Estimate derivatives
            ww_slice = WW[:, sum(ncell[:indtr[jt]]):sum(ncell[:indtr[jt]+1])]

            M_product = ww_slice @ M
            X1M = X1 @ M.T

            der[jt] = ((X1M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

        # Collect derivatives
        iaux = 0
        for jt in range(len(indtr)):
            YY[:, iaux:iaux + der[jt].shape[1]] = der[jt] # construct derivative
            J[jiter] += Jadd[jt]
            iaux += der[jt].shape[1]
            transportMap[indtr[jt]] = tMap[jt]

        # Solve for A using regression
        Anew = np.zeros_like(A)

        for jg in range(ndim):

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

        if disp == 'all':
            print(f"Iteration {jiter+1}/{iterations} done.")

    # Final A is the unregularized one
    A = Anew

    if disp != 'off':
         print("Model identification complete.")

    #OUTPUT = [XX, YY, transportMap, J, A, D, WW, corNet, indw, indexp, TFflag, difs, out, opts]
    return  A

def GRIT_MATLAB_NO_SVD_VarianceNormalization(scdata,Tgrid,epsilon):

    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference
    WITHOUT branching, zero weight and TF flag. 
    WITHOUT SVD, normalization of variance 

    Parameters:
    -----------
    scdata : list of numpy arrays
        Single-cell data for each time point
    Tgrid : list or numpy array
        Time points for the data
    opts : dict, optional
        Options for the algorithm
    Returns:
    --------
    Gene regulatory matrix A and the transport maps
    size of A is (n_genes, n_genes + 1)
    """

    disp = 'off'
    iterations = 30

    # Get dimensions
    ndim = scdata[0].shape[0]  # Number of genes
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

    # Process time grid for Single experiment
    Tgrid = np.array(Tgrid)
    indtr = np.arange(len(scdata)-1)
    indw = np.arange(sum(ncell[:-1]))
    Ttot = max(Tgrid) - min(Tgrid)

    # Check consistency
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # # at each time point, calculate variance for each gene over all cells
    # vvs = np.zeros((ndim, len(indtr)))
    # for jt in range(len(indtr)):
    #     mtemp = np.mean(scdata[indtr[jt]+1], axis=1, keepdims=True) 
    #     mtemp[np.isnan(mtemp)] = 0
    #     vvs[:, jt] += np.sum((scdata[indtr[jt]+1]- mtemp)**2, axis=1)
    #     vvs[:, jt] = vvs[:, jt] / ncell[indtr[jt]+1] 

    # # weight the vvs by number of cells in the previous time step 
    # weighted_vvs = np.zeros_like(vvs)
    # for jt in range(len(indtr)):
    #     weighted_vvs[:, jt] = ncell[indtr[jt]] * vvs[:, jt]

    # # final variance is 
    # vvs = 0.5 + 0.2 * np.sum(weighted_vvs, axis=1, keepdims=True) / np.sum(ncell[indtr]) + 0.8 * vvs
    # vvs = vvs**(-0.5)

    # Build regression matrices
    XX = np.empty((ndim + nbr, 0))
    #XX = np.empty((ndim, 0)) #change 1
    DT = np.empty(0)

    for jt in range(len(indtr)):
        branch_padding = np.ones((1, ncell[indtr[jt]])) 
        data_with_branch = np.vstack([scdata[indtr[jt]],branch_padding]) #change 2
        XX = np.hstack([XX, data_with_branch])
        #data_at_jt = scdata[indtr[jt]]
        #XX = np.hstack([XX, data_at_jt])

        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])

    YY = np.zeros((ndim, XX.shape[1]))

    # Regularization weights
    D = 0.01 * Ttot * np.diag(XX @ XX.T) / XX.shape[1]
    D[-nbr:] = 10 * D[-nbr:] #change 3

    #  Weight matrix for each cell, set to 1
    WW = np.ones((ndim, sum(ncell)))

    # Scale XX by time differences
    XX = DT * XX

    A = np.zeros((ndim, ndim+nbr)) # change 4
    #A = np.ones((ndim, ndim))
    difs = np.zeros(iterations)
    J = np.zeros(iterations)

    uFin = [1] * len(indtr)
    transportMap = [None] * len(scdata)

    # Prepare for main iteration loop
    convergenceProblemIndicator = [0] * len(indtr)
    tMap = [None] * len(indtr)
    Jadd = [0] * len(indtr)
    its = [0] * len(indtr)
    der = [None] * len(indtr)

    # Main iterative loop
    for jiter in range(iterations):
        kreg = min(0.5 + 0.7 * (jiter+1) / iterations, 1)
        Aold = A.copy()

        # Process each transition
        for jt in range(len(indtr)):
            # Propagated and target points
            # change 5
            branch_padding = np.ones((1, ncell[indtr[jt]])) 
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ np.vstack([scdata[indtr[jt]], branch_padding])
            #X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]
            X1 = scdata[indtr[jt]+1]

            # # Cost matrix
            ett0 = np.ones((X0.shape[1], 1))
            ett1 = np.ones((X1.shape[1], 1))


            # vX0 = vvs[:, jt].reshape(-1, 1) * X0
            # vX1 = vvs[:, jt].reshape(-1, 1) * X1

            # ignore svd
            #UvX0 = Ured.T @ vX0
            #UvX1 = Ured.T @ vX1

            # ignore svd
            UvX0 = X0
            UvX1 = X1

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
            epsloc = epsilon
            failInd = True
            failedSinkhornIterations = -1

            M = None
            uFinal = None
            transport_cost = 0
            reg_cost = 0
            iteration_count = 0

            while failInd and failedSinkhornIterations < 10:
                transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, epsloc * np.median(C), uInit)
                #transport_plan = ot.sinkhorn(a, b, cost_matrix, epsilon, method=method)

                failInd = np.sum(np.isnan(M)) > 0
                epsloc = 1.5 * epsloc
                failedSinkhornIterations += 1

            if jiter == iterations - 1 and failedSinkhornIterations > 0:
                convergenceProblemIndicator[jt] = 1

            tMap[jt] = M
            uFin[jt] = uFinal
            Jadd[jt] = transport_cost + epsilon * np.median(C) * reg_cost
            its[jt] = iteration_count

            M = M / np.sum(M, axis=1, keepdims=True)

            # Estimate derivatives
            ww_slice = WW[:, sum(ncell[:indtr[jt]]):sum(ncell[:indtr[jt]+1])]

            M_product = ww_slice @ M
            X1M = X1 @ M.T

            der[jt] = ((X1M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

        # Collect derivatives
        iaux = 0
        for jt in range(len(indtr)):
            YY[:, iaux:iaux + der[jt].shape[1]] = der[jt] # construct derivative
            J[jiter] += Jadd[jt]
            iaux += der[jt].shape[1]
            transportMap[indtr[jt]] = tMap[jt]

        # Solve for A using regression
        Anew = np.zeros_like(A)

        for jg in range(ndim):

            # WV = np.zeros(YY.shape[1])
            # iaux = 0
            # for jt in range(len(indtr)):
            #     WV[iaux:iaux + ncell[indtr[jt]]] = vvs[jg, jt]**2
            #     iaux += ncell[indtr[jt]]

            # Weight matrices
            # ignore variance weight
            # W_diag = WV * WW[jg, indw]
            W_diag = np.ones(YY.shape[1]) * WW[jg, indw]
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

        if disp == 'all':
            print(f"Iteration {jiter+1}/{iterations} done.")

    # Final A is the unregularized one
    A = Anew

    if disp != 'off':
         print("Model identification complete.")

    #OUTPUT = [XX, YY, transportMap, J, A, D, WW, corNet, indw, indexp, TFflag, difs, out, opts]
    return  A

def GRIT_MATLAB_NO_SVD_VarianceNormalization_kreg1(scdata,Tgrid,epsilon):

    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference
    WITHOUT branching, zero weight and TF flag. 
    WITHOUT SVD, normalization of variance, kreg =1

    Parameters:
    -----------
    scdata : list of numpy arrays
        Single-cell data for each time point
    Tgrid : list or numpy array
        Time points for the data
    opts : dict, optional
        Options for the algorithm
    Returns:
    --------
    Gene regulatory matrix A and the transport maps
    size of A is (n_genes, n_genes + 1)
    """

    disp = 'off'
    iterations = 30

    # Get dimensions
    ndim = scdata[0].shape[0]  # Number of genes
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

    # Process time grid for Single experiment
    Tgrid = np.array(Tgrid)
    indtr = np.arange(len(scdata)-1)
    indw = np.arange(sum(ncell[:-1]))
    Ttot = max(Tgrid) - min(Tgrid)

    # Check consistency
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # # at each time point, calculate variance for each gene over all cells
    # vvs = np.zeros((ndim, len(indtr)))
    # for jt in range(len(indtr)):
    #     mtemp = np.mean(scdata[indtr[jt]+1], axis=1, keepdims=True) 
    #     mtemp[np.isnan(mtemp)] = 0
    #     vvs[:, jt] += np.sum((scdata[indtr[jt]+1]- mtemp)**2, axis=1)
    #     vvs[:, jt] = vvs[:, jt] / ncell[indtr[jt]+1] 

    # # weight the vvs by number of cells in the previous time step 
    # weighted_vvs = np.zeros_like(vvs)
    # for jt in range(len(indtr)):
    #     weighted_vvs[:, jt] = ncell[indtr[jt]] * vvs[:, jt]

    # # final variance is 
    # vvs = 0.5 + 0.2 * np.sum(weighted_vvs, axis=1, keepdims=True) / np.sum(ncell[indtr]) + 0.8 * vvs
    # vvs = vvs**(-0.5)

    # Build regression matrices
    XX = np.empty((ndim + nbr, 0))
    #XX = np.empty((ndim, 0)) #change 1
    DT = np.empty(0)

    for jt in range(len(indtr)):
        branch_padding = np.ones((1, ncell[indtr[jt]])) 
        data_with_branch = np.vstack([scdata[indtr[jt]],branch_padding]) #change 2
        XX = np.hstack([XX, data_with_branch])
        #data_at_jt = scdata[indtr[jt]]
        #XX = np.hstack([XX, data_at_jt])

        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])

    YY = np.zeros((ndim, XX.shape[1]))

    # Regularization weights
    D = 0.01 * Ttot * np.diag(XX @ XX.T) / XX.shape[1]
    D[-nbr:] = 10 * D[-nbr:] #change 3

    #  Weight matrix for each cell, set to 1
    WW = np.ones((ndim, sum(ncell)))

    # Scale XX by time differences
    XX = DT * XX

    A = np.zeros((ndim, ndim+nbr)) # change 4
    #A = np.ones((ndim, ndim))
    difs = np.zeros(iterations)
    J = np.zeros(iterations)

    uFin = [1] * len(indtr)
    transportMap = [None] * len(scdata)

    # Prepare for main iteration loop
    convergenceProblemIndicator = [0] * len(indtr)
    tMap = [None] * len(indtr)
    Jadd = [0] * len(indtr)
    its = [0] * len(indtr)
    der = [None] * len(indtr)

    # Main iterative loop
    for jiter in range(iterations):
        kreg = 1
        Aold = A.copy()

        # Process each transition
        for jt in range(len(indtr)):
            # Propagated and target points
            # change 5
            branch_padding = np.ones((1, ncell[indtr[jt]])) 
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ np.vstack([scdata[indtr[jt]], branch_padding])
            #X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]
            X1 = scdata[indtr[jt]+1]

            # # Cost matrix
            ett0 = np.ones((X0.shape[1], 1))
            ett1 = np.ones((X1.shape[1], 1))


            # vX0 = vvs[:, jt].reshape(-1, 1) * X0
            # vX1 = vvs[:, jt].reshape(-1, 1) * X1

            # ignore svd
            #UvX0 = Ured.T @ vX0
            #UvX1 = Ured.T @ vX1

            # ignore svd
            UvX0 = X0
            UvX1 = X1

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
            epsloc = epsilon
            failInd = True
            failedSinkhornIterations = -1

            M = None
            uFinal = None
            transport_cost = 0
            reg_cost = 0
            iteration_count = 0

            while failInd and failedSinkhornIterations < 10:
                transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, epsloc * np.median(C), uInit)
                #transport_plan = ot.sinkhorn(a, b, cost_matrix, epsilon, method=method)

                failInd = np.sum(np.isnan(M)) > 0
                epsloc = 1.5 * epsloc
                failedSinkhornIterations += 1

            if jiter == iterations - 1 and failedSinkhornIterations > 0:
                convergenceProblemIndicator[jt] = 1

            tMap[jt] = M
            uFin[jt] = uFinal
            Jadd[jt] = transport_cost + epsilon * np.median(C) * reg_cost
            its[jt] = iteration_count

            M = M / np.sum(M, axis=1, keepdims=True)

            # Estimate derivatives
            ww_slice = WW[:, sum(ncell[:indtr[jt]]):sum(ncell[:indtr[jt]+1])]

            M_product = ww_slice @ M
            X1M = X1 @ M.T

            der[jt] = ((X1M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

        # Collect derivatives
        iaux = 0
        for jt in range(len(indtr)):
            YY[:, iaux:iaux + der[jt].shape[1]] = der[jt] # construct derivative
            J[jiter] += Jadd[jt]
            iaux += der[jt].shape[1]
            transportMap[indtr[jt]] = tMap[jt]

        # Solve for A using regression
        Anew = np.zeros_like(A)

        for jg in range(ndim):

            # WV = np.zeros(YY.shape[1])
            # iaux = 0
            # for jt in range(len(indtr)):
            #     WV[iaux:iaux + ncell[indtr[jt]]] = vvs[jg, jt]**2
            #     iaux += ncell[indtr[jt]]

            # Weight matrices
            # ignore variance weight
            # W_diag = WV * WW[jg, indw]
            W_diag = np.ones(YY.shape[1]) * WW[jg, indw]
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

        if disp == 'all':
            print(f"Iteration {jiter+1}/{iterations} done.")

    # Final A is the unregularized one
    A = Anew

    if disp != 'off':
         print("Model identification complete.")

    #OUTPUT = [XX, YY, transportMap, J, A, D, WW, corNet, indw, indexp, TFflag, difs, out, opts]
    return  A, transportMap

def GRIT_MATLAB_NO_SVD_VarianceNormalization_kreg1_EpsilonInOT(scdata,Tgrid,epsilon):

    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference
    WITHOUT branching, zero weight and TF flag. 
    WITHOUT SVD, normalization of variance, kreg =1, scaling in OT is epsilon instead of epsilon * median(C)

    Parameters:
    -----------
    scdata : list of numpy arrays
        Single-cell data for each time point
    Tgrid : list or numpy array
        Time points for the data
    epsilon : float
        Regularization parameter for optimal transport
    Returns:
    --------
    Gene regulatory matrix A and the transport maps
    size of A is (n_genes, n_genes)
    """

    disp = 'off'
    iterations = 30

    # Get dimensions
    ndim = scdata[0].shape[0]  # Number of genes
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

    # Process time grid for Single experiment
    Tgrid = np.array(Tgrid)
    indtr = np.arange(len(scdata)-1)
    indw = np.arange(sum(ncell[:-1]))
    Ttot = max(Tgrid) - min(Tgrid)

    # Check consistency
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # # at each time point, calculate variance for each gene over all cells
    # vvs = np.zeros((ndim, len(indtr)))
    # for jt in range(len(indtr)):
    #     mtemp = np.mean(scdata[indtr[jt]+1], axis=1, keepdims=True) 
    #     mtemp[np.isnan(mtemp)] = 0
    #     vvs[:, jt] += np.sum((scdata[indtr[jt]+1]- mtemp)**2, axis=1)
    #     vvs[:, jt] = vvs[:, jt] / ncell[indtr[jt]+1] 

    # # weight the vvs by number of cells in the previous time step 
    # weighted_vvs = np.zeros_like(vvs)
    # for jt in range(len(indtr)):
    #     weighted_vvs[:, jt] = ncell[indtr[jt]] * vvs[:, jt]

    # # final variance is 
    # vvs = 0.5 + 0.2 * np.sum(weighted_vvs, axis=1, keepdims=True) / np.sum(ncell[indtr]) + 0.8 * vvs
    # vvs = vvs**(-0.5)

    # Build regression matrices
    XX = np.empty((ndim + nbr, 0))
    # XX = np.empty((ndim, 0)) #change 1
    DT = np.empty(0)

    for jt in range(len(indtr)):
        branch_padding = np.ones((1, ncell[indtr[jt]])) 
        data_with_branch = np.vstack([scdata[indtr[jt]],branch_padding]) #change 2
        XX = np.hstack([XX, data_with_branch])
        #data_at_jt = scdata[indtr[jt]]
        #XX = np.hstack([XX, data_at_jt])

        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])

    YY = np.zeros((ndim, XX.shape[1]))

    # Regularization weights
    D = 0.01 * Ttot * np.diag(XX @ XX.T) / XX.shape[1]
    D[-nbr:] = 10 * D[-nbr:] #change 3

    #  Weight matrix for each cell, set to 1
    WW = np.ones((ndim, sum(ncell)))

    # Scale XX by time differences
    XX = DT * XX

    A = np.zeros((ndim, ndim+nbr)) # change 4
    #A = np.ones((ndim, ndim))
    difs = np.zeros(iterations)
    J = np.zeros(iterations)

    uFin = [1] * len(indtr)
    transportMap = [None] * len(scdata)

    # Prepare for main iteration loop
    convergenceProblemIndicator = [0] * len(indtr)
    tMap = [None] * len(indtr)
    Jadd = [0] * len(indtr)
    its = [0] * len(indtr)
    der = [None] * len(indtr)

    # Main iterative loop
    for jiter in range(iterations):
        kreg = 1
        Aold = A.copy()

        # Process each transition
        for jt in range(len(indtr)):
            # Propagated and target points
            # change 5
            branch_padding = np.ones((1, ncell[indtr[jt]])) 
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ np.vstack([scdata[indtr[jt]], branch_padding])
            #X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]
            X1 = scdata[indtr[jt]+1]

            # # Cost matrix
            ett0 = np.ones((X0.shape[1], 1))
            ett1 = np.ones((X1.shape[1], 1))


            # vX0 = vvs[:, jt].reshape(-1, 1) * X0
            # vX1 = vvs[:, jt].reshape(-1, 1) * X1

            # ignore svd
            #UvX0 = Ured.T @ vX0
            #UvX1 = Ured.T @ vX1

            # ignore svd
            UvX0 = X0
            UvX1 = X1

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
            epsloc = epsilon
            failInd = True
            failedSinkhornIterations = -1

            M = None
            uFinal = None
            transport_cost = 0
            reg_cost = 0
            iteration_count = 0

            while failInd and failedSinkhornIterations < 10:
                transport_cost, reg_cost, M, iteration_count, uFinal =  OTsolver_MATLABCODE(mu0, mu1, C, epsloc, uInit)
                #transport_plan = ot.sinkhorn(a, b, cost_matrix, epsilon, method=method)

                failInd = np.sum(np.isnan(M)) > 0
                epsloc = 1.5 * epsloc
                failedSinkhornIterations += 1

            if jiter == iterations - 1 and failedSinkhornIterations > 0:
                convergenceProblemIndicator[jt] = 1

            tMap[jt] = M
            uFin[jt] = uFinal
            Jadd[jt] = transport_cost + epsilon * np.median(C) * reg_cost
            its[jt] = iteration_count

            M = M / np.sum(M, axis=1, keepdims=True)

            # Estimate derivatives
            ww_slice = WW[:, sum(ncell[:indtr[jt]]):sum(ncell[:indtr[jt]+1])]

            M_product = ww_slice @ M
            X1M = X1 @ M.T

            der[jt] = ((X1M / M_product) - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

        # Collect derivatives
        iaux = 0
        for jt in range(len(indtr)):
            YY[:, iaux:iaux + der[jt].shape[1]] = der[jt] # construct derivative
            J[jiter] += Jadd[jt]
            iaux += der[jt].shape[1]
            transportMap[indtr[jt]] = tMap[jt]

        # Solve for A using regression
        Anew = np.zeros_like(A)

        for jg in range(ndim):

            # WV = np.zeros(YY.shape[1])
            # iaux = 0
            # for jt in range(len(indtr)):
            #     WV[iaux:iaux + ncell[indtr[jt]]] = vvs[jg, jt]**2
            #     iaux += ncell[indtr[jt]]

            # Weight matrices
            # ignore variance weight
            # W_diag = WV * WW[jg, indw]
            W_diag = np.ones(YY.shape[1]) * WW[jg, indw]
            
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

        if disp == 'all':
            print(f"Iteration {jiter+1}/{iterations} done.")

    # Final A is the unregularized one
    A = Anew

    if disp != 'off':
         print("Model identification complete.")

    #OUTPUT = [XX, YY, transportMap, J, A, D, WW, corNet, indw, indexp, TFflag, difs, out, opts]
    return  A, transportMap

def GRIT_MATLAB_No_adhock_nob(scdata, Tgrid, epsilon):
    """
    Python implementation of GRITmodelSelect function for gene regulatory network inference
    The only adhock thing is when OT solver fails, update epsilon by 1.1*epsilon
    no regularization in the regression step
    b=0 
    
    Parameters:
    -----------
    scdata : list of numpy arrays
        Single-cell data for each time point
        (time x number of genes x number of cells)
    Tgrid : list or numpy array
        Time points for the data
    epsilon : float
        Regularization parameter for optimal transport
    Returns:
    --------
    A: size of A is (n_genes, n_genes)
        Gene regulatory matrix A and the transport maps
    transportMap: list of ncell by ncell numpy arrays with length len(scdata). The last entry is None.
        Transport maps for each time point for last iteration
    """

    # Set display mode and number of iterations
    disp = 'off'  # Controls output verbosity during execution
    iterations = 30  # Number of iterations for convergence

    # Get the number of genes from the first time point data
    ndim = scdata[0].shape[0]  # Number of genes

    # Initialize array to store cell counts at each time point
    ncell = np.zeros(len(scdata), dtype=int)
    # nzeros = 0  # Commented out: counter for zero entries (unused)
    # nelements = 0  # Commented out: counter for total elements (unused)

    # Count cells at each time point
    for jt in range(len(scdata)):
        ncell[jt] = scdata[jt].shape[1]  # Number of cells at time point jt
        # nzeros += np.sum(scdata[jt] == 0)  # Commented out: count zeros (unused)
        # nelements += scdata[jt].size  # Commented out: count total elements (unused)

    # Convert time grid to numpy array and get transition indices
    Tgrid = np.array(Tgrid)
    indtr = np.arange(len(scdata)-1)  # Indices for transitions between time points

    # Check that time grid length matches data length
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # Initialize regression matrices
    XX = np.empty((ndim, 0))  # Matrix to store gene expression data
    DT = np.empty(0)  # Vector to store time differences

    # Build regression matrices by concatenating data from all time points
    for jt in range(len(indtr)):
        data_at_jt = scdata[indtr[jt]]  # Get data at current time point
        XX = np.hstack([XX, data_at_jt])  # Add data to XX matrix
        
        # Calculate square root of time differences for each cell
        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])  # Add time differences to DT vector

    # Initialize target matrix for regression (will hold derivatives)
    YY = np.zeros((ndim, XX.shape[1]))

    # Scale gene expression data by time differences
    XX = DT * XX

    # Initialize gene regulatory matrix A with ones (instead of zeros in the previous version)
    # This means initially all genes are assumed to regulate all other genes
    A = np.ones((ndim, ndim))
    difs = np.zeros(iterations)  # Track differences between iterations

    # Initialize array to store transport maps between time points
    transportMap = [None] * len(scdata)

    # Initialize arrays for main iteration loop
    tMap = [None] * len(indtr)  # Temporary storage for transport maps
    der = [None] * len(indtr)  # Derivatives

    # Main iterative optimization loop
    for jiter in range(iterations):
        Aold = A.copy()  # Store previous A for tracking convergence

        # Process each transition between consecutive time points
        for jt in range(len(indtr)):
            # Propagate current state to next time point using current A matrix
            # Note: No branch padding used in this version
            X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]  # Propagated data
            X1 = scdata[indtr[jt]+1]  # Target data

            # Create ones vectors for cost matrix calculation
            ett0 = np.ones((X0.shape[1], 1))
            ett1 = np.ones((X1.shape[1], 1))

            # Skip SVD decomposition
            UvX0 = X0
            UvX1 = X1

            # Calculate squared norms for each cell
            sum_UvX0_squared = np.sum(UvX0**2, axis=0).reshape(-1, 1)
            sum_UvX1_squared = np.sum(UvX1**2, axis=0)

            # Calculate cost matrix C using squared Euclidean distances
            # C_{ij} = ||x_i||^2 + ||y_j||^2 - 2x_i^T y_j
            C = sum_UvX0_squared @ ett1.T - 2 * UvX0.T @ UvX1 + ett0 @ sum_UvX1_squared.reshape(1, -1)

            # Create uniform distributions for source and target cells
            mu0 = np.ones(np.shape(X0)[1])/np.shape(X0)[1]
            mu1 = np.ones(np.shape(X1)[1])/np.shape(X1)[1]

            # Set up for optimal transport problem
            epsloc = epsilon  # Local regularization parameter
            failInd = True  # Flag for Sinkhorn convergence failure
            failedSinkhornIterations = -1  # Counter for failed iterations

            M = None  # Transport matrix

            # Try solving OT problem with increasing regularization if needed
            while failInd and failedSinkhornIterations < 10:
                # Solve optimal transport problem using external function
                # Note: Not capturing uFinal, iteration_count or costs (simplified from previous version)
                _, _, M, _, _ = OTsolver_MATLABCODE(mu0, mu1, C, epsloc)
                
                # Check if solution contains NaN values
                failInd = np.sum(np.isnan(M)) > 0
                # epsloc = 1.1 * epsloc  # Increase regularization (less aggressively than previous version)
                failedSinkhornIterations += 1
            # print(f"<><> FINAL EPSILON IN SINKHORN WITH EPS: {epsloc}, FAILED: {failInd}<><>")

            # Store transport map for this transition
            tMap[jt] = M

            # Normalize transport map rows to sum to 1
            M = M / np.sum(M, axis=1, keepdims=True)

            # Estimate derivatives using transport map
            # Note: Simpler formula without weighted product
            X1M = X1 @ M.T
            der[jt] = (X1M - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

        # Collect derivatives from all transitions
        iaux = 0
        for jt in range(len(indtr)):
            YY[:, iaux:iaux + der[jt].shape[1]] = der[jt]  # Store derivatives in YY
            iaux += der[jt].shape[1]  # Update index
            transportMap[indtr[jt]] = tMap[jt]  # Store transport map

        # Solve for A using regression (no regularization)
        Anew = np.zeros_like(A)

        for jg in range(ndim):
            # Standard least squares regression without regularization
            reg_matrix = XX @ XX.T  # No regularization term added to diagonal
            Anew[jg, :] = YY[jg] @ XX.T @ np.linalg.inv(reg_matrix)

        # Update A (no regularization mixing parameter, directly use new solution)
        A = Anew

        # Track convergence by measuring change in A
        difs[jiter] = np.sqrt(np.sum((A - Aold)**2))

        # Display progress if requested
        if disp == 'all':
            print(f"Iteration {jiter+1}/{iterations} done.")

    # Final A is the same as the last iteration
    A = Anew

    # Print completion message if not in 'off' mode
    if disp != 'off':
         print("Model identification complete.")
         
    # Return the gene regulatory matrix A and transport maps
    return A, transportMap

def GRIT_MATLAB_regression_step_only(scdata, Tgrid, transportMap):
    """
    Regression step of GRIT algorithm without optimal transport
    
    Parameters:
    -----------
    scdata : list of numpy arrays
        Single-cell data for each time point
    Tgrid : list or numpy array
        Time points for the data
    transportMap : list of ncell by ncell numpy arrays with length len(scdata)-1 (number of tansition)
        Transport maps for each transition between time points
    Returns:
    --------
    Gene regulatory matrix A and the transport maps
    size of A is (n_genes, n_genes)
    """

    # Get the number of genes from the first time point data
    ndim = scdata[0].shape[0]  # Number of genes

    # Initialize array to store cell counts at each time point
    ncell = np.zeros(len(scdata), dtype=int)
    # nzeros = 0  # Commented out: counter for zero entries (unused)
    # nelements = 0  # Commented out: counter for total elements (unused)

    # Count cells at each time point
    for jt in range(len(scdata)):
        ncell[jt] = scdata[jt].shape[1]  # Number of cells at time point jt

    # Convert time grid to numpy array and get transition indices
    Tgrid = np.array(Tgrid)
    indtr = np.arange(len(scdata)-1)  # Indices for transitions between time points

    # Check that time grid length matches data length
    if len(Tgrid) != len(scdata):
        raise ValueError("Time grid vector length does not match the data structure size")

    # Initialize regression matrices
    XX = np.empty((ndim, 0))  # Matrix to store gene expression data
    DT = np.empty(0)  # Vector to store time differences

    # Build regression matrices by concatenating data from all time points
    for jt in range(len(indtr)):
        data_at_jt = scdata[indtr[jt]]  # Get data at current time point
        XX = np.hstack([XX, data_at_jt])  # Add data to XX matrix
        
        # Calculate square root of time differences for each cell
        dt_values = (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5 * np.ones(ncell[indtr[jt]])
        DT = np.hstack([DT, dt_values])  # Add time differences to DT vector

    # Initialize target matrix for regression (will hold derivatives)
    YY = np.zeros((ndim, XX.shape[1]))

    # Scale gene expression data by time differences
    XX = DT * XX

    # Initialize gene regulatory matrix A with ones (instead of zeros in the previous version)
    # This means initially all genes are assumed to regulate all other genes
    A = np.ones((ndim, ndim))

    # Initialize arrays for main iteration loop
    der = [None] * len(indtr)  # Derivatives

# Process each transition between consecutive time points
    for jt in range(len(indtr)):
        # Propagate current state to next time point using current A matrix
        # Note: No branch padding used in this version
        X0 = scdata[indtr[jt]] + (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]]) * A @ scdata[indtr[jt]]  # Propagated data
        X1 = scdata[indtr[jt]+1]  # Target data

        M =transportMap[jt]  # define transport map

        # Normalize transport map rows to sum to 1
        M = M / np.sum(M, axis=1, keepdims=True)

        # Estimate derivatives using transport map
        # Note: Simpler formula without weighted product
        X1M = X1 @ M.T
        der[jt] = (X1M - scdata[indtr[jt]]) / (Tgrid[indtr[jt]+1] - Tgrid[indtr[jt]])**0.5

    # Collect derivatives from all transitions
    iaux = 0
    for jt in range(len(indtr)):
        YY[:, iaux:iaux + der[jt].shape[1]] = der[jt]  # Store derivatives in YY
        iaux += der[jt].shape[1]  # Update index

    # Solve for A using regression (no regularization)
    Anew = np.zeros_like(A)

    for jg in range(ndim):
        # Standard least squares regression without regularization
        reg_matrix = XX @ XX.T  # No regularization term added to diagonal
        Anew[jg, :] = YY[jg] @ XX.T @ np.linalg.inv(reg_matrix)

    # Update A (no regularization mixing parameter, directly use new solution)
    A = Anew
         
    # Return the gene regulatory matrix A and transport maps
    return A

def generate_sparse_A(n_genes, sparsity=0.4):
    """
    Generate a sparse A matrix where:
    - sparsity% of entries are 0
    - The rest are either 1 or -1 with equal probability
    
    Args:
        n_features: Size of the matrix (n_features x n_features)
        sparsity: Fraction of entries that should be zero (default 0.4)
    
    Returns:
        A: The generated sparse matrix
    """
    # Create a matrix of random values between 0 and 1
    mask = np.random.random((n_genes, n_genes))
    
    # Set entries to 0 where mask is less than sparsity
    zeros = mask < sparsity
    
    # For the rest, randomly assign 1 or -1
    ones_or_neg_ones = np.random.choice([-1, 1], size=(n_genes, n_genes))
    
    # Apply the mask
    A = np.where(zeros, 0, ones_or_neg_ones)
    
    return A

def generate_matrix_with_specific_eigenvalues(n_genes, eig_type):
    """
    Generate a 2x2 matrix with specific eigenvalue characteristics
    
    Parameters:
    -----------
    n_genes: int
        Dimension of the matrix (2 for this analysis)
    eig_type: str
        Type of eigenvalues ('positive', 'negative', 'imaginary')
        
    Returns:
    --------
    A: ndarray
        2x2 matrix with the specified eigenvalue characteristics
    """
    if n_genes != 2:
        raise ValueError("This function currently only supports 2x2 matrices")
    
    if eig_type == 'positive':
        # Create a matrix with positive eigenvalues
        eig1 = np.random.uniform(0.1, 1.0)
        eig2 = np.random.uniform(0.1, 1.0)
        
        # Create random eigenvectors
        P = np.random.randn(n_genes, n_genes)
        # Ensure P is invertible
        while np.linalg.det(P) < 1e-10:
            P = np.random.randn(n_genes, n_genes)
            
        # Create diagonal matrix with eigenvalues
        D = np.diag([eig1, eig2])
        
        # Create matrix A = P⁻¹DP
        A = np.linalg.inv(P) @ D @ P
        
    elif eig_type == 'negative':
        # Create a matrix with negative eigenvalues
        eig1 = np.random.uniform(-1.0, -0.1)
        eig2 = np.random.uniform(-1.0, -0.1)
        
        # Create random eigenvectors
        P = np.random.randn(n_genes, n_genes)
        # Ensure P is invertible
        while np.linalg.det(P) < 1e-10:
            P = np.random.randn(n_genes, n_genes)
            
        # Create diagonal matrix with eigenvalues
        D = np.diag([eig1, eig2])
        
        # Create matrix A = P⁻¹DP
        A = np.linalg.inv(P) @ D @ P
        
    elif eig_type == 'imaginary':
        # Create a matrix with complex conjugate eigenvalues
        real_part = np.random.uniform(-0.5, 0.5)
        imag_part = np.random.uniform(0.1, 1.0)
        
        # For a 2x2 real matrix to have complex eigenvalues, it needs to have form:
        # [a, b]
        # [c, d]
        # Where (a+d)² < 4bc to ensure complex roots
        
        a = real_part
        d = real_part
        b = imag_part
        c = -imag_part
        
        A = np.array([[a, b], [c, d]])
    
    else:
        raise ValueError("eig_type must be one of: 'positive', 'negative', 'imaginary'")
    
    # Ensure A is real
    A = np.real(A)
    
    # Verify eigenvalues
    eigvals = np.linalg.eigvals(A)
    
    return A, eigvals