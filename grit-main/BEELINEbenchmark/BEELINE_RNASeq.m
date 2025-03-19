
%Make sure to download the BEELINE data and check the path specifications
%below for where to find the data.

transformations = {'noTransformation','sqrt','log'};

for jtrans = 1:3

    pathname = '../../BEELINE/BEELINE-data/inputs/scRNA-Seq/';
    addpath('..')

    networkpathname = '../../BEELINE/BEELINE-Networks/';
    cases = {'hESC','hHep','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L'};

    %p-value cutoff used for selecting TFs. Value 0.01 is used in BEELINE
    pvalCutoff = 0.01;

    %Number of time points in case they are artificially generated (only mHSC)
    NT = 6;

    %Gene selections to run
    sizesToRun = 1:2;
    if jtrans == 3
        sizesToRun = 1:4;
    end

    %Make a directory to save results
    resdir = ['./Results/results_' transformations{jtrans}];
    mkdir(resdir)

    AUCS = cell(7,4);
    compTimes = zeros(7,4);
    dims = zeros(7,4);
    for jcase = 1:7

        for jsize = sizesToRun

            %%%%%%   SPECIFY CASE DETAILS   %%%%%%

            if jcase < 2.5
                species = 'human';
            else
                species = 'mouse';
            end

            Ngenes = 500 + 500*mod(jsize+1,2);
            includeTFs = jsize < 2.5;
            clear('scdata')


            %%%%%%   READ DATA AND CHOOSE GENES TO INCLUDE   %%%%%%

            %Read expression data
            TT = readtable([pathname cases{jcase} '/ExpressionData.csv']);

            %Read transcription factor list
            TFtable = readtable([networkpathname species '-tfs.csv']);
            TFs = TFtable.TF;

            %Read gene variability table
            GVARtable = readtable([pathname cases{jcase} '/GeneOrdering.csv']);

            %Truncate table to significant genes with Bonferroni correction
            nsign = sum(GVARtable.VGAMpValue < pvalCutoff/size(TT,1));
            GVARtable = GVARtable(1:nsign,:);

            %Sort by variance
            GVARtable = sortrows(GVARtable,'Variance','descend');

            %Create a flag vector for the gene table for included genes and
            %another for TFs
            includeGene = false(nsign,1);
            TFflag = false(nsign,1);

            %Find TFs and check if they are all included
            for jg = 1:nsign
                TFflag(jg) = sum(strcmpi(GVARtable.Var1{jg},TFs)) > 0; %Note: case-insensitive comparison
                if includeTFs
                    includeGene(jg) = TFflag(jg);
                end
            end

            %Include further genes specified by Ngenes
            includeGene(1:sum(cumsum(1-includeGene) <= Ngenes)) = true;

            %Truncate the gene list to included genes only
            GVARtable = GVARtable(includeGene,:);
            TFflag = TFflag(includeGene);

            %Truncate the data table accordingly and order it in the same order
            %as GVARtable
            indAux = zeros(size(GVARtable,1),1);
            for jg = 1:size(GVARtable,1)
                indAux(jg) = find(strcmpi(GVARtable.Var1{jg},TT.Var1)); %Note: case-insensitive comparison
            end
            TT = TT(indAux,:);
            geneNames = TT.Var1;
            dims(jcase,jsize) = length(geneNames);




            %%%%%%   FORM DATA STRUCTURE   %%%%%%

            %Put data into measured time points (done differently for different
            %datasets)
            if jcase == 1 || jcase == 3 || jcase == 4
                %The sampling times are obtained from the cell identifiers
                vv = TT.Properties.VariableNames;
                vv = vv(2:end);
                tn = zeros(1,length(vv));

                for jc = 1:length(vv)
                    iaux = [];
                    iend = [];
                    for jl = 1:length(vv{jc})
                        if strcmp(vv{jc}(jl),'_')
                            iaux = [iaux jl];
                        end   
                        if strcmp(vv{jc}(jl),'h')
                            iend = jl-1;
                        end
                    end
                    istart = iaux(sum(iaux<iend))+1;
                    tn(jc) = str2double(vv{jc}(istart:iend));
                end
                Tgrid = unique(tn);

                %Collect different time points data into a cell structure
                %(recall that 1st column of TT contains gene names)
                 for jt = 1:length(Tgrid)
                     scdata{jt} = table2array(TT(:,[-1 tn] == Tgrid(jt)));
                 end

            end

            %In hHEP case, the cell identifiers contain just cell types.
            %Different measurement times can be deduced from pseudotime.
            if jcase == 2
                Tgrid = [0 6 8 14 21];
                scdata{1} = table2array(TT(:,266:345));
                scdata{2} = table2array(TT(:,2:71));
                scdata{3} = table2array(TT(:,72:184));
                scdata{4} = table2array(TT(:,185:265));
                scdata{5} = table2array(TT(:,346:426));
            end

            %In case of mHSC data (cases 5,6,7), time points are artificially
            %generated based on the pseudotime sorting
            if jcase > 4.5
                PTtable = readtable([pathname cases{jcase} '/PseudoTime.csv']);
                [PT,sortInd] = sort(PTtable.PseudoTime,'ascend');
                TT = [TT(:,1) TT(:,sortInd+1)];
                nn = (size(TT,2)-1)/NT;

                iaux = 0;
                for jt = 1:NT
                    scdata{jt} = table2array(TT(:,iaux+2:round(jt*nn)+1));           
                    Tgrid(jt) = mean(PT(iaux+1:round(jt*nn)));
                    iaux = iaux + size(scdata{jt},2);
                end

            end




            %%%%%%  GROUND TRUTH NETWORKS  %%%%%%


            clear('GT')
            %STRING network
            GTtable = readtable([networkpathname 'Networks/' species '/STRING-network.csv']);
            GT{1} = GTtable2array(GTtable,upper(geneNames),true);

            %Non-specific ChIP-Seq
            if jcase < 2.5
                GTtable = readtable([networkpathname 'Networks/' species '/Non-specific-ChIP-seq-network.csv']);
            else
                GTtable = readtable([networkpathname 'Networks/' species '/Non-Specific-ChIP-seq-network.csv']);
            end
            GT{2} = GTtable2array(GTtable,upper(geneNames),true);

            %Cell-type specific ChIP-Seq
            if jcase < 4.5
                if jcase == 2
                    GTtable = readtable([networkpathname 'Networks/' species '/HepG2-ChIP-seq-network.csv']);
                else
                    GTtable = readtable([networkpathname 'Networks/' species '/' cases{jcase} '-ChIP-seq-network.csv']);
                end
                GT{3} = GTtable2array(GTtable,upper(geneNames),true);
            else
                GTtable = readtable([networkpathname 'Networks/mouse/mHSC-ChIP-seq-network.csv']);
                GT{3} = GTtable2array(GTtable,geneNames,true);
            end

            %Additional ground truth network for mESC case
            if jcase == 4
                GTtable = readtable([networkpathname 'Networks/mouse/mESC-lofgof-network.csv']);
                GT{4} = GTtable2array(GTtable,upper(geneNames),true);
            end

            %Report network statistics as per BEELINE article (Fig. 5 & Suppl.
            %Fig. 8)
            disp(repmat('*',1,30))
            disp(['Case: ' cases{jcase} ', ' num2str(Ngenes) ', Include TFs: ' num2str(includeTFs)])
            netNames = {'STRING','Non-specific ChIP-Seq','Type-specific ChIP-Seq','lof/gof'};
            for jnet = 1:length(GT)
                ntf = sum(sum(GT{jnet},1) > 0);
                nover = sum((sum(GT{jnet},2)>0)'.*(sum(GT{jnet},1)>0));
                ng = sum(sum(GT{jnet},2) > 0) + ntf - nover;
                density = sum(GT{jnet}(:))/(ntf*ng - ntf);
                disp([netNames{jnet} ' -- No. genes: ' num2str(ng) ', No. TFs: ' num2str(ntf) ', density: ' num2str(density)])
            end
            disp(repmat('*',1,30))       
            
            %Transformations
            if jtrans == 1
                for jt = 1:length(scdata)
                    scdata{jt} = 2.^scdata{jt}-1;
                end
            elseif jtrans == 2
                for jt = 1:length(scdata)
                    scdata{jt} = (2.^scdata{jt}-1).^.5;
                    
                end
            end
            for jt = 1:length(scdata)
                ss = sum(scdata{jt},1);
                indsIncl = (ss-mean(ss))/var(ss)^.5 < 4;
                scdata{jt} = scdata{jt}(:,indsIncl);
            end
            

            %%%%%%  RUN METHOD  %%%%%%
            opts = struct;
            opts.par = 12;  
            tic
            [netPred,A,transportMap,~,~,out] = GRIT(scdata,Tgrid,[],[],opts);
            compTimes(jcase,jsize) = toc;

            savefilename = ['results_case' num2str(jcase) '_size' num2str(jsize) '.mat'];
            save([resdir '/' savefilename],'netPred','J0','A','transportMap')



            %%%%%%  PERFORMANCE  %%%%%%
            netPred = netPred-min(netPred(:));

            AUCS{jcase,jsize} = zeros(length(GT),3);
            parpool(length(GT));
            parfor jnet = 1:length(GT)

                %Only include genes on the output side that have an ingoing    
                %link, and genes on the input side that have an outgoing link.
                %Exclude diagonal elements outside the Performance-function
                %since they are no longer diagonal when the matrices are
                %truncated.
                outincl = sum(GT{jnet},2) > 0;
                inincl = sum(GT{jnet},1) > 0;
                indincl = logical((outincl*inincl).*(1-eye(length(outincl))));
                gt = GT{jnet}(indincl);
                pred = netPred(indincl);
                [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(gt,pred,false);
                AUCtemp{jnet} = [AUROC AUPR EPR];
            end
            delete(gcp('nocreate'))
            for jnet = 1:length(GT)
                AUCS{jcase,jsize}(jnet,:) = AUCtemp{jnet};
            end

        end

    end

    save(['Results/RNASeq_AUCS_' transformations{jtrans} '.mat'],'AUCS','compTimes','vars')
end