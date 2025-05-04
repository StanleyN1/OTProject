
%This file runs the experiment for the branch labelling and plots the
%confidence matrix shown in the paper.


%Make sure the BEELINE data can be found in the path below.
addpath('..')
pathname = '../../BEELINE/BEELINE-data/inputs/Synthetic/';

cases = {'dyn-BF','dyn-BFC','dyn-CY','dyn-LI','dyn-LL','dyn-TF'};

%Cases are:
% dyn-BF: bifurcating
% dyn-BFC: bifurcating - converging
% dyn-CY: cycling
% dyn-LI: linear
% dyn-LL: linear long
% dyn-TF: trifurcating


Ncells = [100 200 500 2000 5000];

NT = 5;

jcase = 6;
jsim = 5;
        
dataInd = floor(jsim/5.1) + 1;
jsize = mod(jsim-1,5)+1;
Nc = Ncells(jsize);
foldername = ['/' cases{jcase} '-' num2str(Nc) '-' num2str(dataInd) '/'];
caseInfo((jcase-1)*50 + jsim,1:3) = [jcase dataInd Nc];

%Read data and pseudotime 
TT = readtable([pathname cases{jcase} foldername  'ExpressionData.csv']);
geneNames = TT.Var1;

opts = detectImportOptions([pathname cases{jcase} foldername 'PseudoTime.csv']);
for jvar = 2:length(opts.VariableTypes)
    opts.VariableTypes{jvar} = 'double';
end
PTtable = readtable([pathname cases{jcase} foldername 'PseudoTime.csv'],opts);
if size(PTtable,2) > 2
    PT = table2array(PTtable(:,2:end));
else
    PT = PTtable.PseudoTime;
end
PTtable = readtable([pathname cases{jcase} foldername 'PseudoTime.csv']);
if size(PTtable,2) > 2
    PT = str2double(table2array(PTtable(:,2:end)));
else
    PT = PTtable.PseudoTime;
end

%Read the ground truth
GTtable = readtable([pathname cases{jcase} foldername 'refNetwork.csv']);
GT = GTtable2array(GTtable,geneNames);
Y = table2array(TT(:,2:end));
ndim = size(Y,1);

        
%Create NT different time points artificially by splitting the
%data into equally sized parts
clear('scdata','branchId','Tgrid')
[PTsingle,sortInd] = sort(nansum(PT,2),'ascend');
PT = PT(sortInd,:);
Y = Y(:,sortInd);
nn = size(Y,2)/NT;
iaux = 0;
for jt = 1:NT
    scdata{jt} = Y(:,iaux+1:round(jt*nn));
    Tgrid(jt) = mean(PTsingle(iaux+1:round(jt*nn)));
    branchId{jt} = double(~isnan(PT(iaux+1:round(jt*nn),:)'));
    iaux = iaux + size(scdata{jt},2);
end

[netPred,A0,transportMap] = GRIT(scdata,Tgrid,[],[]);


%%

%Run the code above with jcase = 6, jsim = 5 first

%To have results comparable with Slingshot branch assignments, a high
%threshold value (1) is used to ensure each cell is assigned to exactly one
%branch. The Slingshot branch labels for the last timepoint are used as
%cluster labels required by branchLabeler.
bid = branchLabeler(transportMap,branchId{end},1);

%Create the confusion matrix between identified branch labels and Slingshot
%labels
CM = cell(1,length(transportMap));
for jt = 1:length(transportMap)
    CM{jt} = zeros(size(bid{1},1),size(bid{1},1));
    ii = length(bid)-jt;
    for jcell = 1:size(bid{ii},2)
        CM{jt}(find(bid{ii}(:,jcell)>0),find(branchId{ii}(:,jcell)>0)) = CM{jt}(find(bid{ii}(:,jcell)>0),find(branchId{ii}(:,jcell)>0))+1;
    end
end


%save('conf.mat','CM')

%% 2x2 panel arrangement


%load('conf.mat')

ts = {'T_3','T_2','T_1','T_0','interpreter','latex'};
figure('Position',[300 250 450 360])
for jt = 1:4  
    subplot(2,2,jt)
    confusionchart(CM{5-jt})
    ylabel('Slingshot assignment')
    xlabel('GRIT assignment')
    title(ts{5-jt})
end

%% 1x4 panel arrangement

%load('conf.mat')

figure('Position',[100 100 900 200])
for jt = 1:4  
    subplot(1,4,jt)
    confusionchart(CM{5-jt})
    ylabel('Slingshot assignment')
    xlabel('GRIT assignment')
    title(ts{5-jt})
end


