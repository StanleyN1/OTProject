
addpath('..')
addpath('./dataGeneration')
Ntps = [6 3 12];
epsilon = 0.260368018765957;
[A,b,GT,~,~] = getLinSystem(1,false);

%Code for the experiment presented in Section 2.2 "Performance on linear discrete-time dynamics"

for jtp = 1:3

    %Number of time points
    simOpts.T = Ntps(jtp);

    %Folder for saving results
    resdir = ['Results/Experiment_1_results_' num2str(simOpts.T) 'timepoints'];
    mkdir(resdir)

    %Number of cells per time point
    ncs = [1000 2000 3000 5000 7500 10000];

    %Initialise result matrices
    compTimes = zeros(6,5);
    AUROC_GRN = zeros(6,5);
    AUPR_GRN = zeros(6,5);
    AUROC_A = zeros(6,5);
    AUPR_A = zeros(6,5);
    Aerror = zeros(6,5);
    AerrorMin = zeros(6,5);

    for jnc = 1:length(ncs)

        for jrep = 1:5

            %Generate the data with fixed random seed
            simOpts.seed = jrep;

            %Time difference between measurements 
            simOpts.d = .4;

            %Cells per time point
            simOpts.nc = ncs(jnc);

            %Initial spread
            simOpts.Ks = .35;

            %Noise intensity
            simOpts.Kn = 1/50^.5;

            %Continuous or discrete time simulations
            simOpts.cont = false;

            %Time step
            simOpts.dt = .01;

            %Noise type
            simOpts.noisetype = 'uniform';

            %Simulate data
            [scdata,Tgrid] = linData(simOpts);

            %Run method (OMTMod file has Tikhonov regularisation switched
            %off, that is, lambda_A = lambda_b = 0, and entropy
            %regularisation parameter forced to known noise intensity).
            %Dimension reduction in the OT problem is also swithced off
            %from the options below, by setting opts.Nred to the system
            %dimension (10).
            tic
            opts = struct;
            opts.Nred = 10;
            opts.iterations = 40;
            opts.par = 12;
            [netPred,Aest,transportMap,J,~,out] = GRITmod(scdata,Tgrid,[],[],opts);
            compTimes(jnc,jrep) = toc;


            %Check performance
            [AUROC_GRN(jnc,jrep), AUPR_GRN(jnc,jrep), TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
            [AUROC_A(jnc,jrep), AUPR_A(jnc,jrep), TPR, FPR, PREC, CONF, EPR] = Performance(GT,abs(Aest(:,1:10)),true,false);
            
            Aerror(jnc,jrep) = sum((Aest(:)-[A(:); b]).^2);
            AerrorMin(jnc,jrep) = min(out.error);
            
            savefilename = ['results_size' num2str(jnc) '_rep' num2str(jrep) '.mat'];
            save([resdir '/' savefilename],'Aest','J','out')

            save([resdir '/Experiment_1_results_' num2str(simOpts.T) 'timepoints.mat'],'AUROC_GRN','AUPR_GRN','AUROC_A','AUPR_A','Aerror','AerrorMin','compTimes')            

        end
    end   
end
            
 

