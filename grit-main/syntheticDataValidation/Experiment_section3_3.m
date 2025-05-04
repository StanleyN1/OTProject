%Code for the experiment presented in Section 2.4 "Perturbation target inference"


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

[~,~,GT,~,~] = getLinSystem(1,false);
clear('perturbations')


% 60% increase in basal transription rate of one gene

pertRes1 = zeros(10,20);
corRes1 = zeros(10,20);
GTpert1 = zeros(10,20);
for jrep = 1:20
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    rng(jrep)
    indpert = randperm(10,1);
    pertC = ones(10,1);
    pertC(indpert) = 1.6;
    GTpert1(indpert,jrep) = 1;
    
    [scdataM,TgridM] = MutData(simOpts,[],[1;1;1],pertC);
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Perturbation = {[],[1]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change summed up over lines
    corRes1(:,jrep) = corChange(scdataC,scdataM,0);
    
    %Store results
    pertRes1(:,jrep) = GRN(:,11);
    disp(['Case 1, replicate ' num2str(jrep) ' finished.'])
end


% 60% increase in basal transription rate of two genes

pertRes2 = zeros(10,20);
corRes2 = zeros(10,20);
GTpert2 = zeros(10,20);
for jrep = 1:20
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    rng(jrep)
    indpert = randperm(10,2);
    pertC = ones(10,1);
    pertC(indpert) = 1.6;
    GTpert2(indpert,jrep) = 1;
    
    [scdataM,TgridM] = MutData(simOpts,[],[1;1;1],pertC);
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Perturbation = {[],[1]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change summed up over lines
    corRes2(:,jrep) = corChange(scdataC,scdataM,0);
    
    %Store results
    pertRes2(:,jrep) = GRN(:,11);
    disp(['Case 2, replicate ' num2str(jrep) ' finished.'])
end


% 60% increase in basal transription rate of three genes

pertRes3 = zeros(10,20);
corRes3 = zeros(10,20);
GTpert3 = zeros(10,20);
for jrep = 1:20
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    rng(jrep)
    indpert = randperm(10,3);
    pertC = ones(10,1);
    pertC(indpert) = 1.6;
    GTpert3(indpert,jrep) = 1;
    
    [scdataM,TgridM] = MutData(simOpts,[],[1;1;1],pertC);
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Perturbation = {[],[1]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change summed up over lines
    corRes3(:,jrep) = corChange(scdataC,scdataM,0);
    
    %Store results
    pertRes3(:,jrep) = GRN(:,11);
    disp(['Case 3, replicate ' num2str(jrep) ' finished.'])
end


% 30% increase in basal transription rate of one gene

pertRes4 = zeros(10,20);
corRes4 = zeros(10,20);
GTpert4 = zeros(10,20);
for jrep = 1:20
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    rng(jrep)
    indpert = randperm(10,1);
    pertC = ones(10,1);
    pertC(indpert) = 1.3;
    GTpert4(indpert,jrep) = 1;
    
    [scdataM,TgridM] = MutData(simOpts,[],[1;1;1],pertC);
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Perturbation = {[],[1]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change summed up over lines
    corRes4(:,jrep) = corChange(scdataC,scdataM,0);
    
    %Store results
    pertRes4(:,jrep) = GRN(:,11);
    disp(['Case 4, replicate ' num2str(jrep) ' finished.'])
end


% 30% increase in basal transription rate of two genes

pertRes5 = zeros(10,20);
corRes5 = zeros(10,20);
GTpert5 = zeros(10,20);
for jrep = 1:20
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    rng(jrep)
    indpert = randperm(10,2);
    pertC = ones(10,1);
    pertC(indpert) = 1.3;
    GTpert5(indpert,jrep) = 1;
    
    [scdataM,TgridM] = MutData(simOpts,[],[1;1;1],pertC);
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Perturbation = {[],[1]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change summed up over lines
    corRes5(:,jrep) = corChange(scdataC,scdataM,0);
    
    %Store results
    pertRes5(:,jrep) = GRN(:,11);
    disp(['Case 5, replicate ' num2str(jrep) ' finished.'])
end


% 30% increase in basal transription rate of three genes

pertRes6 = zeros(10,20);
corRes6 = zeros(10,20);
GTpert6 = zeros(10,20);
for jrep = 1:20
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    rng(jrep)
    indpert = randperm(10,3);
    pertC = ones(10,1);
    pertC(indpert) = 1.3;
    GTpert6(indpert,jrep) = 1;
    
    [scdataM,TgridM] = MutData(simOpts,[],[1;1;1],pertC);
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Perturbation = {[],[1]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change summed up over lines
    corRes6(:,jrep) = corChange(scdataC,scdataM,0);
    
    %Store results
    pertRes6(:,jrep) = GRN(:,11);
    disp(['Case 6, replicate ' num2str(jrep) ' finished.'])
end

save('Results/Experiment3_results.mat','pertRes1','pertRes2','pertRes3','pertRes4','pertRes5','pertRes6','corRes1','corRes2','corRes3','corRes4','corRes5','corRes6','GTpert1','GTpert2','GTpert3','GTpert4','GTpert5','GTpert6')

