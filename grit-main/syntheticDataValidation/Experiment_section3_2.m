
%Code for the experiment presented in Section 2.3 "Effect of continuous time, nonlinear dynamics, and hidden states"

%% Set simulation options

addpath('..')
addpath('./dataGeneration')

%Time difference between measurements
simOpts.d = .4;

%Number of time points
simOpts.T = 6;

%Cells per time point
simOpts.nc = 600;

%Initial spread
simOpts.Ks = .35;

%Noise intensity
simOpts.Kn = 1/50^.5;

%Time step
simOpts.dt = .01;

simOpts.noisetype = 'uniform';

AUCS_GRN = zeros(8,20);
AUCS_A = zeros(8,20);
AUCS_Bulk = zeros(8,20);
[A,b,GT,~,~] = getLinSystem(1,false);

%% Linear discrete-time systems

%Continuous or discrete time simulations
simOpts.cont = false;

for jrep = 1:20
    simOpts.seed = jrep;
    [scdata,Tgrid] = linData(simOpts);

    %Run the method
    [netPred,Aest,transportMap,J,~,out] = GRIT(scdata,Tgrid,[],[]);

    %Solve linear regression on the bulk data
    clear('bulkdata')
    for jt = 1:length(scdata)
        bulkdata{jt} = mean(scdata{jt},2)*ones(1,size(scdata{jt},2));
    end
    [bulkPred] = GRIT(bulkdata,Tgrid,[],[]);

    %Check performance
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
    AUCS_GRN(1:2,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,abs(Aest(:,1:10)),true,false);
    AUCS_A(1:2,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,bulkPred,true,false);
    AUCS_Bulk(1:2,jrep) = [AUROC; AUPR];

    disp(['Batch 1, replicate ' num2str(jrep) ' finished.'])
end


% Linear continuous-time systems

%Continuous or discrete time simulations
simOpts.cont = true;

for jrep = 1:20
    simOpts.seed = jrep;
    [scdata,Tgrid] = linData(simOpts);

    %Run the method
    [netPred,Aest,transportMap,J,~,out] = GRIT(scdata,Tgrid,[],[]);

    %Solve linear regression on the bulk data
    clear('bulkdata')
    for jt = 1:length(scdata)
        bulkdata{jt} = mean(scdata{jt},2)*ones(1,size(scdata{jt},2));
    end
    [bulkPred] = GRIT(bulkdata,Tgrid,[],[]);

    %Check performance
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
    AUCS_GRN(3:4,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,abs(Aest(:,1:10)),true,false);
    AUCS_A(3:4,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,bulkPred,true,false);
    AUCS_Bulk(3:4,jrep) = [AUROC; AUPR];

    disp(['Batch 2, replicate ' num2str(jrep) ' finished.'])
end

% Non-Linear continuous-time systems

for jrep = 1:20
    simOpts.seed = jrep;
    [scdata,Tgrid] = NonLinData(simOpts);

    %Run the method
    [netPred,Aest,transportMap,J,~,out] = GRIT(scdata,Tgrid,[],[]);

    %Solve linear regression on the bulk data
    clear('bulkdata')
    for jt = 1:length(scdata)
        bulkdata{jt} = mean(scdata{jt},2)*ones(1,size(scdata{jt},2));
    end
    [bulkPred] = GRIT(bulkdata,Tgrid,[],[]);

    %Check performance
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
    AUCS_GRN(5:6,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,abs(Aest(:,1:10)),true,false);
    AUCS_A(5:6,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,bulkPred,true,false);
    AUCS_Bulk(5:6,jrep) = [AUROC; AUPR];

    disp(['Batch 3, replicate ' num2str(jrep) ' finished.'])
end

% Non-Linear continuous-time systems with protein dynamics

for jrep = 1:20
    simOpts.seed = jrep;
    [scdata,Tgrid] = proteinData(simOpts);

   %Run the method
    [netPred,Aest,transportMap,J,~,out] = GRIT(scdata,Tgrid,[],[]);

    %Solve linear regression on the bulk data
    clear('bulkdata')
    for jt = 1:length(scdata)
        bulkdata{jt} = mean(scdata{jt},2)*ones(1,size(scdata{jt},2));
    end
    [bulkPred] = GRIT(bulkdata,Tgrid,[],[]);

    %Check performance
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
    AUCS_GRN(7:8,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,abs(Aest(:,1:10)),true,false);
    AUCS_A(7:8,jrep) = [AUROC; AUPR];
    
    [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,bulkPred,true,false);
    AUCS_Bulk(7:8,jrep) = [AUROC; AUPR];

    disp(['Batch 4, replicate ' num2str(jrep) ' finished.'])
end

save('Results/Experiment_2_results.mat','AUCS_GRN','AUCS_A','AUCS_Bulk')

