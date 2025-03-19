%% Load data

addpath('/Users/atte.aalto/Documents/MATLAB/LRRK2data/')

filenames{1} = 'GSM3660746_LRRK2_WT_0_Days.txt';
filenames{2} = 'GSM3660746_LRRK2_WT_10_Days.txt';
filenames{3} = 'GSM3660746_LRRK2_WT_14_Days.txt';
filenames{4} = 'GSM3660746_LRRK2_WT_42_Days.txt';
filenames{5} = 'GSM3660745_LRRK2_G2019S_0_Days.txt';
filenames{6} = 'GSM3660745_LRRK2_G2019S_10_Days.txt';
filenames{7} = 'GSM3660745_LRRK2_G2019S_14_Days.txt';
filenames{8} = 'GSM3660745_LRRK2_G2019S_42Days.txt';

%% Form the gene list from those genes that are included in all samples

%Get the gene list
opts = detectImportOptions(filenames{1});
opts.SelectedVariableNames = {'GENE'};
geneList = readtable(filenames{1},opts);
disp(['File 1 processed.'])
for jfile = 2:8

    %Only read the first column
    opts = detectImportOptions(filenames{jfile});
    opts.SelectedVariableNames = {'GENE'};
    T1 = readtable(filenames{jfile},opts); 

    %Include only genes that are included in all time points
    geneList = intersect(geneList,T1);
    disp(['File ' num2str(jfile) ' processed.'])
end
geneList = geneList.GENE;

%Create the transcription factor flag
TFtable = readtable('./BEELINE/BEELINE-networks/human-tfs.csv');
TFs = TFtable.TF;
TFflag = false(length(geneList),1);
for jg = 1:length(geneList)
    if max(strcmp(geneList{jg},TFs)) == 1
        TFflag(jg) = true;
    end
end


%% Calculate nr. of non-zero elements, mean, and variance from cells with at least 1000 reads

nrcell = zeros(1,8);
geneSum = zeros(length(geneList),8);
geneSqSum = zeros(length(geneList),8);
nPos = zeros(length(geneList),8);

for jfile = 1:8
    T1 = readtable(filenames{jfile});
    inds = zeros(length(geneList),1);
    for jg = 1:length(geneList)
        inds(jg) = find(strcmp(geneList{jg},T1.GENE));
    end
    Ytemp = table2array(T1(inds,2:end));
    indcell = sum(Ytemp,1) > 1000;
    nrcell(jfile) = sum(indcell);
    geneSum(:,jfile) = sum(Ytemp(:,indcell),2);
    geneSqSum(:,jfile) = sum(Ytemp(:,indcell).^2,2);
    nPos(:,jfile) = sum(Ytemp(:,indcell) > 0,2);
    
    disp(['File ' num2str(jfile) ' processed.'])
end

inds = find(sum(nPos,2) > .1*sum(nrcell));
geneList = geneList(inds);

%save('LRRK2geneList.mat','geneList','fullGeneList','nrcell')

%% Concatenate gene-list to those that are expressed in at least 10% of cells and form the data matrices

%load('LRRK2geneList.mat')

YC = zeros(length(geneList),sum(nrcell(1:4)));
YM = zeros(length(geneList),sum(nrcell(5:8)));
ranC = [];
ranM = [];
iauxC = 0;
iauxM = 0;
for jfile = 1:8
    T1 = readtable(filenames{jfile});
    inds = zeros(length(geneList),1);
    for jg = 1:length(geneList)
        inds(jg) = find(strcmp(geneList{jg},T1.GENE));
    end

    Ytemp = table2array(T1(inds,2:end));
    indcell = sum(Ytemp,1) > 1000;
    if jfile < 4.5
        YC(:,(1:sum(indcell)) + iauxC) = Ytemp(:,indcell);
        ranC = [ranC, [iauxC+1; iauxC + sum(indcell)]];
        iauxC = iauxC + sum(indcell);
    else
        YM(:,(1:sum(indcell)) + iauxM) = Ytemp(:,indcell);
        ranM = [ranM, [iauxM+1; iauxM + sum(indcell)]];
        iauxM = iauxM + sum(indcell);
    end
    disp(['File ' num2str(jfile) ' processed.'])
end
clear('T1','Ytemp')

%% Normalise by the total count of each cell and multiply by a scale factor

KC = median(sum(YC,1));
KM = median(sum(YM,1));

YC = KC*YC./sum(YC,1);
YM = KM*YM./sum(YM,1);

%% Calculate for each gene the sum of Wasserstein distances over time steps

DG = zeros(size(YC,1),2);
for jg = 1:size(YC,1)
    for jt = 2:size(ranC,2)
        DG(jg,1) = DG(jg,1) + Wass1D(YC(jg,ranC(1,jt):ranC(2,jt)),YC(jg,ranC(1,jt-1):ranC(2,jt-1)));
    end
end

for jg = 1:size(YM,1)
    for jt = 2:size(ranM,2)
        DG(jg,2) = DG(jg,2) + Wass1D(YM(jg,ranM(1,jt):ranM(2,jt)),YM(jg,ranM(1,jt-1):ranM(2,jt-1)));
    end
end

%% Make the dataset from most varying genes and run method

dims = [500 1000 1500 2000];

mkdir('Results')
for jd = 1:length(dims)

    %Generate data structure
    [~,isort] = sort(sum(DG,2),'descend');
    inds = isort(1:dims(jd));
    gNames = geneList(inds);
    clear('scdata','Tgrid')
    
    %Include Control experiment
    for jt = 1:4
        scdata{jt} = YC(inds,ranC(1,jt):ranC(2,jt));
    end
    
    %Include Mutation experiment
    for jt = 1:4
        scdata{4+jt} = YM(inds,ranM(1,jt):ranM(2,jt));
    end
    
    %Time points given as a cell array to indicate multiple experiments
    Tgrid = {[0 10 14 42],[0 10 14 42]};

    %Indicate perturbations
    perturbations = struct;
    perturbations.Perturbation = {[],1};

    %Run method
    opts = struct;
    opts.par = 12;
    [GRN,~,~,~,~,~] = GRITpert(scdata,Tgrid,[],[],perturbations,opts);


    %Put results in a table and save
    [~,isort] = sort(GRN(:,end),'descend');
    Tout = table(geneList(inds(isort)),GRN(isort,end));
    Tout.Properties.VariableNames = {'Gene','Confidence'};
    writetable(Tout,['./Results/LRRK2_dim' num2str(dims(jd)) '.csv'])
    save(['./Results/LRRK2_dim' num2str(dims(jd)) '.mat'],'GRN','gNames')

end




