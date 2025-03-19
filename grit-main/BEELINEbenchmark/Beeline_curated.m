

%Number of timepoints the data is split into
NT = 8;

if exist('NTenforce')
    NT = NTenforce;
    warning(['Number of timepoints enforced to: ' num2str(NT)])
end
addpath('..')
pathname = '../../BEELINE/BEELINE-data/inputs/Curated/';
cases = {'GSD','HSC','mCAD','VSC'};
AUCS = zeros(120,3);
caseInfo = zeros(120,4);
missingN = [0 50 70];
missingS = {'','-50','-70'};

for jcase = 1:4

    for jsim = 1:30
        
        dataInd = floor(jsim/3.01) + 1;
        miss = missingS{mod(jsim-1,3)+1};
        foldername = ['/' cases{jcase} '-2000-' num2str(dataInd) miss '/'];
        caseInfo((jcase-1)*30 + jsim,1:3) = [jcase dataInd missingN(mod(jsim-1,3)+1)];
        
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
        
        %Read ground truth
        GTtable = readtable([pathname cases{jcase} foldername 'refNetwork.csv']);
        GT = GTtable2array(GTtable,geneNames);
        Y = table2array(TT(:,2:end));
        ndim = size(Y,1);
        
        
        %Create NT different time points artificially by splitting the
        %data into four equally sized parts
        clear('scdata','Tgrid','branchId')
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
        [netPred,A,transportMap] = GRIT(scdata,Tgrid,[],branchId);
        
        %Check performance
        caseInfo((jcase-1)*30 + jsim,4) = toc;
        [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,netPred,true,false);
        AUCS((jcase-1)*30 + jsim,:) = [AUROC AUPR EPR];
            
    end
    
    disp(['Case ' num2str(jcase) ' finished.'])
    
end

save(['./Results/curated_results_' num2str(NT) 'timepoints.mat'],'AUCS','caseInfo')


