
%% Generate Synthetic Data
rng(42);
n_genes = 5;
n_timepoints = 30;
n_cells = 50;
epsilon = 1;
% A_true = 0.1*([
%     -0.5, 0.2, 0, 0, 0;
%     0.1, -0.1, 0.3, -0.8, 0;
%     0, 0.1, -0.7, 0.2, 0;
%     0, 0.8, 0.2, -0.8, 0.1;
%     0, 0, 0, 0.1, -0.9]);
A_true = 0.1*eye(n_genes);
b_true = 0.1*ones(n_genes,1);
b_true = zeros(n_genes,1);
x0 = rand(n_genes,1)*5; 

%[scdata, Tgrid] = SyntheticDataGeneration_YQ(A_true, b_true, x0, n_timepoints, n_cells, epsilon,"/Users/yqian46/Library/Mobile Documents/com~apple~CloudDocs/PHD/Research/GRN/GRN Inference Code/Data");
[scdata, Tgrid] = SyntheticDataGeneration_YQ(A_true, b_true, x0, n_timepoints, n_cells, epsilon);
opts = struct;

%% GRIT CODE
TFflag = [];
branchId = [];
opts = struct;

out = struct;

%Check which options are given and set default values for others
if ~isfield(opts,'epsilon')
    opts.epsilon = .05;
end
if ~isfield(opts,'iterations')
    opts.iterations = 30;
end
if ~isfield(opts,'zeroWeight')
    opts.zeroWeight = 1;
end
if ~isfield(opts,'disp')
    opts.disp = 'basic';
end
if ~isfield(opts,'par')
    opts.par = 0;
end
if ~isfield(opts,'Nred')
    opts.Nred = min(100,round(.9*size(scdata{1},1)));
end
if ~isfield(opts,'maxReg')
    opts.maxReg = 40;
end
if ~isfield(opts,'branchWeight')
    opts.branchWeight = 2;
end
if ~isfield(opts,'signed')
    opts.signed = false;
end
if opts.par > 1
    parpool(opts.par);
end

if isempty(branchId)
    for jt = 1:length(scdata)
        branchId{jt} = ones(1,size(scdata{jt},2));
    end
end

%Number of genes
ndim = size(scdata{1},1);

%Number of branches
nbr = size(branchId{1},1);

%Numbers of cells in samples
ncell = zeros(1,length(scdata));
nzeros = 0;
nelements = 0;
for jt = 1:length(scdata)
    ncell(jt) = size(scdata{jt},2);
    nzeros = nzeros + sum(scdata{jt}(:) == 0);
    nelements = nelements + numel(scdata{jt});
    if min(scdata{jt}(:)) < 0 && opts.zeroWeight < 1
        opts.zeroWeight = 1;
        warning('Negative values in the gene expression data. Zero inflation will not be accounted for.')
    end
end

%In case the data consist of several experiments, the Tgrid is a cell array
%corresponding to measurement times of the different experiments.
Ttot = 0;
if iscell(Tgrid)
    Tg = Tgrid;
    clear('Tgrid');
    Tgrid = [];
    indtr = [];
    iaux = 0;
    indw = [];
    indexp = [];
    for jex = 1:length(Tg)
        Tgrid = [Tgrid, Tg{jex}];
        indtr = [indtr, (1:length(Tg{jex})-1) + iaux];
        indw = [indw, sum(ncell(1:iaux)) + (1:sum(ncell((1:length(Tg{jex})-1) + iaux)))];
        indexp = [indexp, jex*ones(1,sum(ncell(iaux+1:iaux+length(Tg{jex})-1)))];
        iaux = iaux + length(Tg{jex});
        Ttot = Ttot + max(Tg{jex})-min(Tg{jex});
    end
else
    indtr = 1:length(scdata)-1;
    indw = 1:sum(ncell(1:end-1));
    Ttot = max(Tgrid)-min(Tgrid);
    indexp = ones(1,sum(ncell(1:end-1)));
end
if length(Tgrid) ~= length(scdata)
    error('Time grid vector length does not match the data structure size')
end

%Transcription factor flag-variable and index-list
TFflag = TFflag(:);
if length(TFflag) < ndim && ~isempty(TFflag)
    warning('Transcription factor input is interpreted as a list of TF indices')
    TFlist = TFflag;
    TFflag = false(ndim,1);
    TFflag(TFlist) = true;
end
if sum(TFflag) == 0
    TFflag = true(ndim,1);
end

%Include the constant load as a "transcription factor"
TFflag = [TFflag; true(nbr,1)];

%If TFflag is given as numeric, it has to be converted to logical
if ~islogical(TFflag)
    TFflag = TFflag > 0;
end


%Calculate variances of each gene at each time point, accounting for
%different branches
vvs = zeros(ndim,length(indtr));
for jt = 1:length(indtr)
    for jbr = 1:nbr
        mtemp = mean(scdata{indtr(jt)+1}(:,branchId{indtr(jt)+1}(jbr,:) > 0),2);
        mtemp(isnan(mtemp)) = 0;
        vvs(:,jt) = vvs(:,jt) + sum((scdata{indtr(jt)+1}(:,branchId{indtr(jt)+1}(jbr,:) > 0) - mtemp).^2,2);
    end
    vvs(:,jt) = vvs(:,jt)/ncell(indtr(jt)+1);
end
vvs = .5+.2*sum(ncell(indtr).*vvs,2)/sum(ncell(indtr))+.8*vvs;
vvs = vvs.^-.5;


%Regression matrices
XX = [];
DT = [];
for jt = 1:length(indtr)
    XX = [XX [scdata{indtr(jt)}; branchId{indtr(jt)}]];
    DT = [DT (Tgrid(indtr(jt)+1)-Tgrid(indtr(jt)))^.5*ones(1,ncell(indtr(jt)))];
end
YY = zeros(ndim,size(XX,2));


%Regularization weights
D = .01*Ttot*diag(XX*XX')/size(XX,2);
D(end-nbr+1:end) = 10*D(end-nbr+1:end);


%Reduced weights for zeros
WW = ones(ndim,sum(ncell));
for jt = 1:length(scdata)
    WW(:,sum(ncell(1:jt-1)) + (1:ncell(jt))) = 1 - (1-opts.zeroWeight).*(scdata{jt} < 1e-10);
end


%Calculate the gene-gene correlations and PC-projector for dimension
%reduction
Xc = XX(1:ndim,:);
indmiss = setdiff(1:length(scdata),indtr);
for im = indmiss
    Xc = [Xc scdata{im}];
end
Xc = Xc - mean(Xc,2);
[Ured,S,~] = svds(Xc,opts.Nred);
corNet = Xc*Xc';
sc = diag(corNet).^-.5;
corNet = sc.*corNet.*sc';
out.vars = diag(S).^2/sum(sum(Xc(:).^2));
indmiss = [0 indmiss];
%clear('Xc')
    

%Scale XX with dt^.5 now that Ured is calculated
XX = DT.*XX;


% Calculate the relative masses in each branch over all time points
branchMass = zeros(size(branchId{1},1),1);
for jt = 1:length(branchId) 
    branchMass = branchMass + sum(branchId{jt},2);
end
branchMass = branchMass/sum(branchMass);
brm = zeros(length(branchMass),length(branchId));
for jt = 1:length(branchId)
    for jb = 1:size(branchId{jt},1)
        brm(jb,jt) = branchMass(jb)/sum(branchId{jt}(jb,:)./sum(branchId{jt},1));   
    end
end
if max(isinf(brm(:)))
    warning('Some branches are not present in all time points!')
end
brm(isinf(brm)) = 1;

% Solve A iteratively
A = ones(ndim,ndim+nbr);
difs = zeros(1,opts.iterations);
J = zeros(1,opts.iterations);
out.its = zeros(opts.iterations,length(indtr));
for jt = 1:length(indtr)
    rat{jt} = 1;
    m_rat = 1;
    uFin{jt} = 1;
end
for jiter = 1:opts.iterations
    kreg = min(.5 + .7*jiter/opts.iterations,1);
    Aold = A;
    
    %parfor (jt = 1:length(indtr), opts.par)
    for jt = 1:length(indtr)
        
        %Propagated and target points
        X0 = scdata{indtr(jt)} + (Tgrid(indtr(jt)+1)-Tgrid(indtr(jt)))*A*[scdata{indtr(jt)}; branchId{indtr(jt)}];
        X1 = scdata{indtr(jt)+1};


        %Cost matrix
        ett0 = ones(size(X0,2),1);
        ett1 = ones(size(X1,2),1);
        C = sum((Ured'*(vvs(:,jt).*X0)).^2,1)'*ett1'- 2*(Ured'*(vvs(:,jt).*X0))'*(Ured'*(vvs(:,jt).*X1)) + ett0*sum((Ured'*(vvs(:,jt).*X1)).^2,1);
        %Increase those elements of the cost that correspond to jumps from
        %a branch to another by a factor opts.branchWeight.
        C = C.*(1 + (opts.branchWeight-1)*(branchId{indtr(jt)}'*branchId{indtr(jt)+1} == 0));
        
        %Discrete mass distributions with modulated masses
        mu0 = sum((branchId{indtr(jt)}./sum(branchId{indtr(jt)},1)).*brm(:,indtr(jt)),1)';
        mu1 = sum((branchId{indtr(jt)+1}./sum(branchId{indtr(jt)+1},1)).*brm(:,indtr(jt)+1),1)';
        mu0 = mu0/sum(mu0);
        mu1 = mu1/sum(mu1);
        if jiter == 1
            uInit = mu1;
        else
            uInit = uFin{jt};
        end
        
        %Solve the OMT problem
        epsloc = opts.epsilon;
        failInd = true;
        failedSinkhornIterations = -1;
        while failInd && failedSinkhornIterations < 10
            [transport_cost,reg_cost, M, iteration_count, uFinal] = OTsolver(mu0,mu1, C, epsloc*median(C(:)), uInit);
            
            failInd = sum(isnan(M(:))) > 0;
            epsloc = 1.5*epsloc;
            failedSinkhornIterations = failedSinkhornIterations + 1;
        end
        if jiter == opts.iterations && failedSinkhornIterations > 0
            convergenceProblemIndicator{jt} = 1;
        else
            convergenceProblemIndicator{jt} = 0;
        end
        tMap{jt} = M;
        uFin{jt} = uFinal;
        M = M./sum(M,2);
        Jadd{jt} = transport_cost + opts.epsilon*median(C(:))*reg_cost;
        its{jt} = iteration_count;
        
        %Estimate derivatives
        der{jt} = ((X1*M')./(WW(:,sum(ncell(1:indtr(jt))) + (1:ncell(indtr(jt)+1)))*M') - scdata{indtr(jt)})/(Tgrid(indtr(jt)+1)-Tgrid(indtr(jt)))^.5;

    end
    iaux = 0;
    for jt = 1:length(indtr)
        YY(:,iaux + (1:size(der{jt},2))) = der{jt};
        J(jiter) = J(jiter) + Jadd{jt};
        iaux = iaux + size(der{jt},2);
        transportMap{indtr(jt)} = tMap{jt};
        out.its(jiter,jt) = its{jt};
    end
    
    %Solve the A-matrix from input-output regression with estimated
    %derivatives
    An = cell(ndim,1);
    Anew = zeros(size(A));
    parfor (jg = 1:ndim, opts.par)
        %Set of regressors are the TFs and the target gene itself
        TFloc = TFflag;
        TFloc(jg) = true;
        WV = zeros(1,size(YY,2));
        iaux = 0;
        for jt = 1:length(indtr)
            WV(iaux + (1:ncell(indtr(jt)))) = vvs(jg,jt)^2;
            iaux = iaux + ncell(indtr(jt));
        end
        An{jg} = (YY(jg,:).*(WV.*WW(jg,indw)))*XX(TFloc,:)'*((XX(TFloc,:).*(WV.*WW(jg,indw)))*XX(TFloc,:)' + diag(D(TFloc)))^-1;
    end
    for jg = 1:ndim
        TFloc = TFflag;
        TFloc(jg) = true;
        Anew(jg,TFloc) = An{jg};
    end
    % Regularise with smaller step size (increasing over iterations)
    A = (1-kreg)*Aold + kreg*Anew;
    
    %Check progress
    difs(jiter) = sum((A(:)-Aold(:)).^2)^.5;
    
    if strcmp(opts.disp,'all')
        disp(['Iteration ' num2str(jiter) '/' num2str(opts.iterations) ' done.'])
    end
end
A = Anew;

%Check for and report about convergence problems
expnr = 0;
for jt = 1:length(convergenceProblemIndicator)
    if convergenceProblemIndicator{jt} == 1
        if expnr == 0
            warning('Convergence problems detected and regularisation increased')
            disp('Check the following matrices for outliers:')
        end
        expnr = 1+indtr(jt)-jt;
        tpnr = indtr(jt)-indmiss(expnr);
        disp(['* Experiment ' num2str(expnr) ', time points ' num2str(tpnr) ' and ' num2str(tpnr+1)])
    end
end

if opts.par > 1
    delete(gcp('nocreate'))
end

if ~strcmp(opts.disp,'off')
    disp('Model identification complete.')
end

%OUPUT = [XX,YY,transportMap,J,A,D,WW,corNet,indw,indexp,TFflag,difs,out,opts];