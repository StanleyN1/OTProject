%Code for the experiment presented in Section 2.5 "Mutation effect inference"

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

clear('perturbations')
[~,~,GT,~,~] = getLinSystem(1,false);

%% 50% loss of function for one pathway

pertRes1 = zeros(10,30);
corRes1 = zeros(10,30);
s = ones(3,30);
s(1,1:10) = .5;
s(2,11:20) = .5;
s(3,21:30) = .5;
for jrep = 1:30
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    
    simOpts.seed = mod(jrep-1,10)+11;
    [scdataM,TgridM] = MutData(simOpts,[],s(:,jrep),ones(10,1));
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Mutation = {[],[9]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change for gene 9
    corRes1(:,jrep) = corChange(scdataC,scdataM,9);
    
    
    %Store results
    pertRes1(:,jrep) = GRN(:,11);
    disp(['Case 1, replicate ' num2str(jrep) ' finished.'])
end

GTmut1 = zeros(10,30);
GTmut1(4,1:10) = 1;
GTmut1(7,11:20) = 1;
GTmut1(10,21:30) = 1;



% 20% loss of function for one pathway

pertRes2 = zeros(10,30);
corRes2 = zeros(10,30);
s = ones(3,30);
s(1,1:10) = .8;
s(2,11:20) = .8;
s(3,21:30) = .8;
for jrep = 1:30
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    simOpts.seed = mod(jrep-1,10)+11;
    [scdataM,TgridM] = MutData(simOpts,[],s(:,jrep),ones(10,1));
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Mutation = {[],[9]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change for gene 9
    corRes2(:,jrep) = corChange(scdataC,scdataM,9);
    
    %Store results
    pertRes2(:,jrep) = GRN(:,11);
    disp(['Case 2, replicate ' num2str(jrep) ' finished.'])
end



% 50% loss of function for two pathways

pertRes3 = zeros(10,30);
corRes3 = zeros(10,30);
s = ones(3,30);
s([2 3],1:10) = .5;
s([1 3],11:20) = .5;
s([1 2],21:30) = .5;
for jrep = 1:30
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    simOpts.seed = mod(jrep-1,10)+11;
    [scdataM,TgridM] = MutData(simOpts,[],s(:,jrep),ones(10,1));
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Mutation = {[],[9]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change for gene 9
    corRes3(:,jrep) = corChange(scdataC,scdataM,9);
    
    %Store results
    pertRes3(:,jrep) = GRN(:,11);
    disp(['Case 3, replicate ' num2str(jrep) ' finished.'])
end

GTmut2 = zeros(10,30);
GTmut2([7 10],1:10) = 1;
GTmut2([4 10],11:20) = 1;
GTmut2([4 7],21:30) = 1;



% 20% loss of function for two pathways

%First link not affected
pertRes4 = zeros(10,30);
corRes4 = zeros(10,30);
s = ones(3,30);
s([2 3],1:10) = .8;
s([1 3],11:20) = .8;
s([1 2],21:30) = .8;
for jrep = 1:30
    %Generate data
    simOpts.seed = mod(jrep-1,10)+1;
    [scdataC,TgridC] = MutData(simOpts,[],[1;1;1],ones(10,1));
    simOpts.seed = mod(jrep-1,10)+11;
    [scdataM,TgridM] = MutData(simOpts,[],s(:,jrep),ones(10,1));
    scdata = {scdataC{1:6}, scdataM{1:6}};
    Tgrid = {TgridC, TgridM};

    %Run method
    perturbations.Mutation = {[],[9]};
    [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,[],[],perturbations);

    %Correlation change for gene 9
    corRes4(:,jrep) = corChange(scdataC,scdataM,9);
    
    %Store results
    pertRes4(:,jrep) = GRN(:,11);
    disp(['Case 4, replicate ' num2str(jrep) ' finished.'])
end

save('Results/Experiment4_results.mat','pertRes1','pertRes2','pertRes3','pertRes4','corRes1','corRes2','corRes3','corRes4','GTmut1','GTmut2')

