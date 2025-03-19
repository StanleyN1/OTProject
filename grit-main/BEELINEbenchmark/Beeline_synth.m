
%Number of timepoints the data is split into
NT = 8;

if exist('NTenforce')
    NT = NTenforce;
    warning(['Number of timepoints enforced to: ' num2str(NT)])
end
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

AUCS = zeros(300,3);
caseInfo = zeros(300,4);
Ncells = [100 200 500 2000 5000];
for jcase = 1:6
    
    for jsim = 1:50
        
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
          
        %Run GRIT
        tic
        [netPred,A0,transportMap] = GRIT(scdata,Tgrid,[],branchId);
        
        %Check performance
        caseInfo((jcase-1)*50 + jsim,4) = toc;
        [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
        AUCS((jcase-1)*50 + jsim,:) = [AUROC AUPR EPR];
      
    end
    
    disp(['Case ' num2str(jcase) ' finished.'])
    
end
save(['./Results/synthetic_results_' num2str(NT) 'timepoints.mat'],'AUCS','caseInfo')
