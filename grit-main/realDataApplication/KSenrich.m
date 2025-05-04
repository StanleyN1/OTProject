

%PINK1
%Tfull = readtable('./Results/gprofilerResults/outFull_PINK1_dim2000_case4.csv');
%Tres = readtable('./Results/PINK1_dim2000_case4.csv');

%Cases with PINK1:
%1 - Treated as mutation dataset, including all time points for control
%2 - Treated as mutation dataset, excluding 3rd and 5th time points
%3 - Treated as perturbation dataset, including all time points for control
%4 - Treated as perturbation dataset, excluding 3rd and 5th time points



%PINK1
Tfull = readtable('./Results/gprofilerResults/PINK1_dim1500_gprof_case4.csv');
Tres = readtable('./Results/PINK1_dim1500_case4.csv');

%LRRK2
%Tfull = readtable('./Results/gprofilerResults/LRRK2_dim1000_gprof.csv');
%Tres = readtable('./Results/LRRK2_dim1000.csv');



TKS = table((1:size(Tfull,1))',Tfull.term_name,zeros(size(Tfull,1),1),zeros(size(Tfull,1),1),zeros(size(Tfull,1),1),zeros(size(Tfull,1),1),zeros(size(Tfull,1),1),zeros(size(Tfull,1),1));
TKS.Properties.VariableNames = {'Rank','Name','p_value','p_adjusted','KS_adjusted','KS','Nr_genes','top200'};
cdfuni = [(1:size(Tres,1))',(1:size(Tres,1))'/size(Tres,1)];

%Go through all KEGG terms on the list
clear('geneAux')
for jlist = 1:size(Tfull,1)

    %Read intersections between KEGG terms and results
    ints = Tfull.intersections(jlist);
    
    %Find spaces in order to parse the list
    ss = strfind(ints{1},',');
    ss = [0 ss length(ints{1})+1];

    %Find the ranks of the genes in the results
    rnk = zeros(1,length(ss)-1);
    for jg = 1:length(ss)-1
        geneAux{jlist,jg} = ints{1}(ss(jg)+1:ss(jg+1)-1);
        rnk(jg) = find(strcmpi(geneAux{jlist,jg},Tres.Gene));
    end

    %Kolmogorov-Smirnov test
    [~,p,ksstat] = kstest(rnk,'CDF',cdfuni,'Tail','larger');
    
    TKS.KS(jlist) = ksstat;
    TKS.KS_adjusted(jlist) = ksstat*length(rnk)^.5;
    TKS.Nr_genes(jlist) = length(rnk);
    TKS.p_value(jlist) = p;     
    TKS.top200(jlist) = round(100*sum(rnk<201)/length(rnk),1);
end

[~,isort] = sort(TKS.p_value,'ascend');
TKS = TKS(isort,:);
TKS.Rank = (1:size(Tfull,1))';
geneAux = geneAux(isort,:);

%Adjusted p-values by correcting for multiple hypothesis testing using
%Benjamini-Hochberg false discovery rate approach (Yoav Benjamini and Yosef
%Hochberg. Controlling the False Discovery Rate: Practical and Powerful 
%Approach to Multiple Testing. Journal of the Royal Statistical Society B, 
%57(1):289-300, 1995.)
TKS.p_adjusted = TKS.p_value*size(TKS,1)./(1:size(TKS,1))';
for jp = 1:size(TKS,1)
    TKS.p_adjusted(jp) = min(TKS.p_adjusted(jp:end));
end





%% Show significant and non-significant with small number of genes (Supp. Table 2 and 3)

nsig = sum(TKS.p_adjusted < .05);
ii = [(1:nsig)';find((TKS.Rank > nsig).*(TKS.p_value < .05).*(TKS.KS > .4).*(TKS.Nr_genes > 1)>0)];

overlaps = zeros(length(ii),length(ii));
for jp1 = 1:length(ii)
    for jp2 = 1:length(ii)
        
        for jg = 1:TKS.Nr_genes(ii(jp1))
            overlaps(jp1,jp2) = overlaps(jp1,jp2) + sum(strcmp(geneAux{ii(jp1),jg},geneAux(ii(jp2),:)));
        end
    end
end
    
%Visualise the table
TKS(ii,:)

%Write the table as a csv file
TKS.p_value = round(TKS.p_value,4,'significant');
TKS.p_adjusted = round(TKS.p_adjusted,4,'significant');
TKS.KS = round(TKS.KS,4,'significant');
TKS.KS_adjusted = round(TKS.KS_adjusted,4,'significant');
%writetable(TKS(ii,:),'Results/EnrichResults_PINK1.csv')


%% For LRRK2, zoom in on the main cluster

%Terms belonging to the main cluster
iCluster = 1:3;

%Show the overlaps of the terms in the cluster. Diagonal shows the term
%sizes.
overlaps(iCluster,iCluster)

%Concatenate all genes in the cluster
clusterGenes = {};
for jterm = iCluster
    clusterGenes = [clusterGenes geneAux(jterm,1:TKS.Nr_genes(jterm))];
end

%For each gene, count the number of terms they belong to
uniqueGenes = unique(clusterGenes);
geneCounter = zeros(1,length(uniqueGenes));
for jgene = 1:length(geneCounter)
    geneCounter(jgene) = sum(strcmp(uniqueGenes{jgene},clusterGenes));
end

%List of genes that belong to all three terms
overlapGenes = uniqueGenes(geneCounter > 2.5);


%Find their ranks in the results
rnk32 = zeros(1,length(overlapGenes));
for jgene = 1:length(rnk32)
    rnk32(jgene) = find(strcmp(overlapGenes{jgene},Tres.Gene));
end


%List of 17 genes in top-400 of GRIT results
topGenes32 = overlapGenes(rnk32 < 400.5);



%% For PINK1, zoom in on the main cluster

%Terms belonging to the main cluster
iCluster = 1:14;

%Show the overlaps of the terms in the cluster. Diagonal shows the term
%sizes.
overlaps(iCluster,iCluster)

%Concatenate all genes in the cluster
clusterGenes = {};
for jterm = iCluster
    clusterGenes = [clusterGenes geneAux(jterm,1:TKS.Nr_genes(jterm))];
end

%For each gene, count the number of terms they belong to
uniqueGenes = unique(clusterGenes);
geneCounter = zeros(1,length(uniqueGenes));
for jgene = 1:length(geneCounter)
    geneCounter(jgene) = sum(strcmp(uniqueGenes{jgene},clusterGenes));
end

%List of genes that belong to at least three terms
overlapGenes = uniqueGenes(geneCounter > 2.5);


%Find their ranks in the results
rnk12 = zeros(1,length(overlapGenes));
for jgene = 1:length(rnk12)
    rnk12(jgene) = find(strcmp(overlapGenes{jgene},Tres.Gene));
end


%List of 17 genes in top-400 of GRIT results
topGenes = overlapGenes(rnk12 < 400.5);

