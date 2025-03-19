function [GRN,A,transportMap,J,difs,out] = GRITmod(scdata,Tgrid,TFflag,branchId,opts)

%NOTE: this is a modified version of GRIT that corresponds exactly to the
%assumptions of the consistency theorem. The modifications include:
% - Tikhonov regularisation parameter is set to zero
% - The entropy regularisation parameter is set to the known noise intensity 

out = struct;

[Atr,btr,~,~,~] = getLinSystem(1,false);
xss = -Atr^-1*btr;
ep = mean((-2*diag(Atr).*xss).^.5);


%Check which options are given and set default values for others
if ~exist('opts')
    opts = struct;
end
if ~isfield(opts,'epsilon')
    opts.epsilon = .05;
end
if ~isfield(opts,'iterations')
    opts.iterations = 30;
end
if ~isfield(opts,'zeroWeight')
    opts.zeroWeight = 0.1;
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
    opts.maxReg = 50;
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
for jt = 1:length(scdata)
    ncell(jt) = size(scdata{jt},2);
    if min(scdata{jt}(:)) < 0 && opts.zeroWeight < 1
        opts.zeroWeight = 1;
        warning('Negative values in the gene expression data. Zero inflation will not be accounted for.')
    end
end

%In case the data consist of several experiments, the Tgrid is a cell array
%corresponding to measurement times of the different experiments.
if iscell(Tgrid)
    Tg = Tgrid;
    clear('Tgrid');
    Tgrid = [];
    indtr = [];
    iaux = 0;
    indw = [];
    nex = length(Tg);
    for jex = 1:length(Tg)
        Tgrid = [Tgrid, Tg{jex}];
        indtr = [indtr, (1:length(Tg{jex})-1) + iaux];
        indw = [indw, sum(ncell(1:iaux)) + (1:sum(ncell((1:length(Tg{jex})-1) + iaux)))];
        iaux = iaux + length(Tg{jex});
        
    end
else
    indtr = 1:length(scdata)-1;
    indw = 1:sum(ncell(1:end-1));
    nex = 1;
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

A = zeros(ndim,ndim+nbr);
difs = zeros(1,opts.iterations);
XX = [];
for jt = 1:length(indtr)
    XX = [XX (Tgrid(indtr(jt)+1)-Tgrid(indtr(jt)))^.5*[scdata{indtr(jt)}; branchId{indtr(jt)}]];
end
YY = zeros(ndim,size(XX,2));


%Regularization weights
D = diag(XX*XX');
D = nex*D/mean(D);
D(end-nbr+1:end) = 10*D(end-nbr+1:end);

%Modification: Tikhonov regularisation switched off
D = D*0;

%Reduced weights for zeros
WW = ones(ndim,sum(ncell));
for jt = 1:length(scdata)
    WW(:,sum(ncell(1:jt-1)) + (1:ncell(jt))) = 1 - (1-opts.zeroWeight)*(scdata{jt} < 1e-10);
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
clear('Xc')
    


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
J = zeros(1,opts.iterations);
out.its = zeros(opts.iterations,length(indtr));
for jt = 1:length(indtr)
    rat{jt} = 1;
    m_rat = 1;
    uFin{jt} = 1;
end
for jiter = 1:opts.iterations
    kreg = .5 + .4*jiter/opts.iterations;
    kreg = min(.5 + .7*jiter/opts.iterations,1);
    Aold = A;
    
    parfor (jt = 1:length(indtr), opts.par)
    %for jt = 1:length(indtr)
        
        %Propagated and target points
        X0 = scdata{indtr(jt)};
        X0 = X0 + (Tgrid(indtr(jt)+1)-Tgrid(indtr(jt)))*A*[X0; branchId{indtr(jt)}];
        X1 = scdata{indtr(jt)+1};

        %Cost matrix
        ett0 = ones(size(X0,2),1);
        ett1 = ones(size(X1,2),1);
        C = sum((Ured'*X0).^2,1)'*ett1'- 2*(Ured'*X0)'*(Ured'*X1) + ett0*sum((Ured'*X1).^2,1);
       
        %Increase those elements of the cost that correspond to jumps from
        %a branch to another by a factor opts.branchWeight.
        C = C.*(1 + (opts.branchWeight-1)*(branchId{indtr(jt)}'*branchId{indtr(jt)+1} == 0));
        
        %Modification: Scale cost with dt
        C = C/.4;
        
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
        
        %Solve the OMT problem (Modification: epsilon fixed to known value)
        [transport_cost,reg_cost, M, iteration_count, uFinal] = OMT_init(mu0, mu1, C, 2*ep^2/50, uInit);
        
        tMap{jt} = M;
        uFin{jt} = uFinal;
        M = M./sum(M,2);
        Jadd{jt} = transport_cost + 2*ep^2/50*reg_cost; %(Modification: epsilon fixed to known value)
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
        An{jg} = (YY(jg,:).*WW(jg,indw))*XX(TFloc,:)'*((XX(TFloc,:).*WW(jg,indw))*XX(TFloc,:)' + diag(D(TFloc)))^-1;
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
    
    
    ee = sum((A(:)-[Atr(:); btr]).^2);
    out.error(jiter) = ee;
    

end
A = Anew;

if ~strcmp(opts.disp,'off')
    disp('Model identification complete.')
end



%=== Network inference ===

maxReg = min(opts.maxReg,sum(TFflag)-nbr);
parfor (jg = 1:ndim, opts.par)
%for jg = 1:ndim
    
    %Initialise the regressor set
    if maxReg == sum(TFflag)-nbr
        TFloc = TFflag;
        fwd = false;
    else
        TFloc = TFflag;
        TFloc(jg) = true;
        [~,indsrt] = sort(TFloc(1:ndim)'.*corNet(jg,:).^2,'descend');
        TFloc = [false(ndim,1); true(nbr,1)];
        TFloc(indsrt(1:maxReg-5)) = true;
        fwd = true;
    end
    
    TFloc(jg) = true;
    TFlocList = find(TFloc)';
    TFleft = and(TFflag,~TFloc);
    TFleftList = find(TFleft);
    TFleftList = TFleftList(:);
    TFlocList = TFlocList(:)';
   
    CF{jg} = zeros(2,ndim);
    %CF{jg}(1,TFflag) = 1;
    CB{jg} = zeros(2,ndim);
    %CB{jg}(1,TFflag) = 1;
    
    while sum(TFloc) > nbr+1 || fwd      
        
        %Solve A (row) with regressors indicated by TFloc
        G1 = ((XX(TFloc,:).*WW(jg,indw))*XX(TFloc,:)' + diag(D(TFloc)))^-1;
        Atemp = (YY(jg,:).*WW(jg,indw))*XX(TFloc,:)'*G1;
        Jnew = sum(WW(jg,indw).*(YY(jg,:) - Atemp*XX(TFloc,:)).^2,2) + sum(Atemp.^2*diag(D(TFloc)),2);
        
        %Switch to backward greedy steps once the forward sweep is complete
        if sum(TFloc)-nbr == maxReg
            fwd = false;
        end
        
        if fwd %Forward greedy step
        
            %Check regressor importance for candidate regressors
            DeltaJ = zeros(1,length(TFleftList));
            for jin = 1:length(TFleftList)
                v = XX(TFleftList(jin),:);
                P22 = 1/(D(TFleftList(jin)) + (v.*WW(jg,indw))*v' - (v.*WW(jg,indw))*XX(TFloc,:)'*G1*XX(TFloc,:)*(v.*WW(jg,indw))');
                P12 = -P22*G1*XX(TFloc,:)*(v.*WW(jg,indw))';
                P11 = G1 + P12*P12'/P22;
                G2 = [P11 P12; P12' P22];
                Atr0 = (YY(jg,:).*WW(jg,indw))*XX([TFlocList, TFleftList(jin)],:)'*G2;
                DeltaJ(jin) = Jnew - sum(WW(jg,indw).*(YY(jg,:) - Atr0*XX([TFlocList, TFleftList(jin)],:)).^2,2) + sum(Atr0.^2*diag(D([TFlocList, TFleftList(jin)])),2);
            end

            %Link scoring
            [Kscale, imax] = max(DeltaJ);
            CF{jg}(1,TFleftList) = CF{jg}(1,TFleftList) + (DeltaJ/Kscale);
            CF{jg}(2,TFleftList) = CF{jg}(2,TFleftList) + 1;
            
            %Add the one with highest importance to the regressor set TFloc
            iToAdd = TFleftList(imax);
            TFleft(iToAdd) = false;
            TFleftList = [TFleftList(1:imax-1); TFleftList(imax+1:end)];
            TFlocList = [TFlocList(1:sum(TFloc(1:iToAdd))) iToAdd  TFlocList(sum(TFloc(1:iToAdd))+1:end)];
            TFloc(iToAdd) = true;
  
        else %Backward greedy step

            %Check regressor importance in the current model
            DeltaJ = zeros(1,length(Atemp)-nbr);
            for jin = 1:length(Atemp)-nbr
                G2 = [G1(1:jin-1,1:jin-1), G1(1:jin-1,jin+1:end); G1(jin+1:end,1:jin-1), G1(jin+1:end,jin+1:end)] - [G1(1:jin-1,jin); G1(jin+1:end,jin)]*[G1(1:jin-1,jin); G1(jin+1:end,jin)]'/G1(jin,jin);               
                Atr0 = (YY(jg,:).*WW(jg,indw))*XX(TFlocList([1:jin-1 jin+1:length(TFlocList)]),:)'*G2;
                Atr0 = [Atr0(1:jin-1) 0 Atr0(jin:end)];
                DeltaJ(jin) = sum(WW(jg,indw).*(YY(jg,:) - Atr0*XX(TFloc,:)).^2,2) + sum(Atr0.^2*diag(D(TFloc)),2) - Jnew;

                if TFlocList(jin) == jg
                    indAux = jin;
                end
            end
                
            %Link scoring
            Kscale = max(DeltaJ);
            CB{jg}(1,TFlocList(1:end-nbr)) = CB{jg}(1,TFlocList(1:end-nbr)) + (DeltaJ/Kscale);
            CB{jg}(2,TFlocList(1:end-nbr)) = CB{jg}(2,TFlocList(1:end-nbr)) + 1;

            %Prevent the diagonal element being excluded
            DeltaJ(indAux) = 1e10;

            %Remove the one with lowest importance from the regressor set TFloc
            [~,imin] = min(DeltaJ);
            iToRemove = TFlocList(imin);

            TFlocList = [TFlocList(1:imin-1) TFlocList(imin+1:end)];
            TFloc(iToRemove) = false;
            TFleftList = [TFleftList(1:sum(TFleft(1:iToRemove))); iToRemove;  TFleftList(sum(TFleft(1:iToRemove))+1:end)];
            TFleft(iToRemove) = true;
     
        end
    end
end
      
%Calculate link scores
GRN = zeros(ndim,ndim);
auxNet = zeros(ndim,ndim);
for jout = 1:ndim
    for jin = 1:ndim
        if CF{jout}(2,jin) > 0
            auxNet(jout,jin) = CF{jout}(1,jin)/CF{jout}(2,jin);
        end
        if CB{jout}(2,jin) > 0
            GRN(jout,jin) = CB{jout}(1,jin)/CB{jout}(2,jin);
        end
    end
end

%Fill in the zeros in GRN using the scores from the forward sweep 
for jg = 1:size(GRN,1)
    zz = GRN(jg,:) == 0;
    GRN(jg,zz) = 0.9*auxNet(jg,zz)/(max(auxNet(jg,zz))+1e-10)*min(GRN(jg,~zz));
end

%Check if signed links are asked for
if opts.signed
    GRN = GRN.*sign(A(:,1:ndim));
end


if opts.par > 1
    delete(gcp('nocreate'))
end

