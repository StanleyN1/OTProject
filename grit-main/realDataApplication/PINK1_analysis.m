%% Load data

addpath('/Users/atte.aalto/Documents/MATLAB/PINK1data/')

TD = readtable('PINK1data.csv');
TM = readtable('PINK1_Metadata.csv');

%% Make sure the cells are in the same order
T1 = TM.Var1;
T2 = TD.Properties.VariableNames;
T2 = T2(2:end);
for jc = 1:length(T1)
    if ~strcmp(T1{jc},T2{jc})
        error('Cell ordering not consistent')
    end
end

%% Separate data into control and mutant

cases = unique(TM.orig_ident);
casesControl = cases([5 1 2 3 4]);
casesMut = cases([9 6 7 8]);

%Create the matrix with control cells
clear('indAux')
for jcase = 1:length(casesControl)
    indAux(jcase,:) = strcmp(TM.orig_ident,casesControl{jcase})';
end
YC0 = zeros(size(TD,1),sum(indAux(:)));
ncellCont = sum(indAux,2);

iAux = 0;
for jcase = 1:length(casesControl)
    YC0(:,(1:ncellCont(jcase)) + iAux) = table2array(TD(:,[false indAux(jcase,:)]));
    iAux = iAux + ncellCont(jcase);
end


%Create the matrix with mutant cells
clear('indAux')
for jcase = 1:length(casesMut)
    indAux(jcase,:) = strcmp(TM.orig_ident,casesMut{jcase})';
end
YM0 = zeros(size(TD,1),sum(indAux(:)));
ncellMut = sum(indAux,2);

iAux = 0;
for jcase = 1:length(casesMut)
    YM0(:,(1:ncellMut(jcase)) + iAux) = table2array(TD(:,[false indAux(jcase,:)]));
    iAux = iAux + ncellMut(jcase);
end

geneNames = TD.Var1;
TgridM = [0 6 15 21];
TgridC = [0 6 10 15 21];
clear('TD','TM')


%% Create index ranges of time points 

ranC = zeros(2,5);
ranM = zeros(2,4);

ia = 0;
for jt = 1:5
    ranC(1,jt) = ia+1;
    ranC(2,jt) = ia+ncellCont(jt);
    ia = ia + ncellCont(jt);
end

ia = 0;
for jt = 1:4
    ranM(1,jt) = ia+1;
    ranM(2,jt) = ia+ncellMut(jt);
    ia = ia + ncellMut(jt);
end



%% Normalise by the total count of each cell and multiply by a scale factor

KC = median(sum(YC0,1));
KM = median(sum(YM0,1));

YC0 = KC*YC0./sum(YC0,1);
YM0 = KM*YM0./sum(YM0,1);


%% transformation

%alp = .085763;

%YC1 = log(1+YC0);
%YM1 = log(1+YM0);

YC1 = YC0; %.^.5;
YM1 = YM0; %.^.5;


%YC1 = acosh(2*alp*YC0+1);
%YM1 = acosh(2*alp*YM0+1);

%% Keep genes that are expressed in more than 10% of cells

thr = .1;

zeroCount = sum([YC1 YM1] == 0,2);
inds = zeroCount < (1-thr)*(size(YC1,2)+size(YM1,2));
YC1 = YC1(inds,:);
YM1 = YM1(inds,:);
geneNames = geneNames(inds);


%% Calculate for each gene the sum of Wasserstein distances over time steps

ctrlCases = [1 2 4];

DG1 = zeros(size(YC1,1),2);
DG2 = zeros(size(YC1,1),2);

%Keeping all time points
for jg = 1:size(YC1,1)
    for jt = 2:size(ranC,2)
        DG1(jg,1) = DG1(jg,1) + Wass1D(YC1(jg,ranC(1,jt):ranC(2,jt)),YC1(jg,ranC(1,jt-1):ranC(2,jt-1)));
    end
end

%Excluding time points 3 and 5
for jg = 1:size(YC1,1)
    for jt = 2:length(ctrlCases)
        DG2(jg,1) = DG2(jg,1) + Wass1D(YC1(jg,ranC(1,ctrlCases(jt)):ranC(2,ctrlCases(jt))),YC1(jg,ranC(1,ctrlCases(jt-1)):ranC(2,ctrlCases(jt-1))));
    end
end



for jg = 1:size(YM1,1)
    for jt = 2:size(ranM,2)
        WassAux = Wass1D(YM1(jg,ranM(1,jt):ranM(2,jt)),YM1(jg,ranM(1,jt-1):ranM(2,jt-1)));
        DG1(jg,2) = DG1(jg,2) + WassAux;
        DG2(jg,2) = DG2(jg,2) + WassAux;
    end
end

%% Make the dataset from most varying genes (and PINK1) and run method
dims = [500 1000 1500 2000];

mkdir('Results')

%Cases:
%1 - Treated as mutation dataset, including all time points for control
%2 - Treated as mutation dataset, excluding 3rd and 5th time points
%3 - Treated as perturbation dataset, including all time points for control
%4 - Treated as perturbation dataset, excluding 3rd and 5th time points

for jcase = 1:4
    for jd = 1:length(dims)

        %Generate data structure
        ndim = dims(jd);

         %Include either all timepoints or only the good timepoints
         if mod(jcase,2) == 1
            ctrlCases = 1:5;
            [~,isort] = sort(sum(DG1,2),'descend');
        else
            ctrlCases = [1 2 4];
            [~,isort] = sort(sum(DG2,2),'descend');
        end

        %Find PINK1 and include it in the dataset if it wasn't there
        indMut = find(strcmp(geneNames,'PINK1'));
        rankMut = find(isort == indMut);
        if rankMut > ndim
            inds = [isort(1:ndim); indMut];
            indMut = ndim + 1;
        else
            inds = isort(1:ndim);
        end
        gNames = geneNames(inds);
        
        clear('scdata','Tgrid')
        
        %Include Control experiment
        for jt = 1:length(ctrlCases)
            scdata{jt} = YC1(inds,ranC(1,ctrlCases(jt)):ranC(2,ctrlCases(jt)));
        end
        
        %Include Mutation experiment
        for jt = 1:4
            scdata{length(ctrlCases)+jt} = YM1(inds,ranM(1,jt):ranM(2,jt));
        end
        
        %Time points given as a cell array to indicate multiple experiments
        Tgrid = {TgridC(ctrlCases), TgridM};
        
        %Indicate perturbations. Try treating the mutation both as a
        %mutation, or an undefined perturbation.
        perturbations = struct;
        if jcase < 2.5
            perturbations.Mutation = {[],indMut};
        else
            perturbations.Perturbation = {[],1};
        end
        

        %Run method
        opts = struct;
        opts.par = 12;
        opts.forcePert = true;
        [GRN,~,~,~,~,~] = GRITpert(scdata,Tgrid,[],[],perturbations,opts);


        %Put results in a table and save
        [~,isort] = sort(GRN(:,end),'descend');
        Tout = table(geneNames(inds(isort)),GRN(isort,end));
        Tout.Properties.VariableNames = {'Gene','Confidence'};
        writetable(Tout,['./Results/PINK1_dim' num2str(ndim) '_case' num2str(jcase) '.csv'])
        save(['./Results/PINK1_dim' num2str(ndim) '_case' num2str(jcase) '.mat'],'GRN','gNames')
        
    end
end


