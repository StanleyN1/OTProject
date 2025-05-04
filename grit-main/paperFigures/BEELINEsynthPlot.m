%% Results with synthetic data compared to methods in BEELINE

opts = detectImportOptions('../../BEELINE/41592_2019_690_MOESM12_ESM.csv');
TT = readtable('../../BEELINE/41592_2019_690_MOESM12_ESM.csv',opts);
methodNames = TT.Var1(1:end);
datasetNames = {'LI','CY','LL','BF','BFC','TF'};
scores = table2array(TT(:,2:7));
methodNames = ['GRIT, 8'; 'GRIT, 15'; methodNames];


%Cases with 2000 or 5000 cells
ii = [4 5 9 10 14 15 19 20 24 25 29 30 34 35 39 40 44 45 49 50];
load('../BEELINEbenchmark/Results/Synth_random_AUPR.mat')

load('../BEELINEbenchmark/Results/synthetic_results_8timepoints.mat')
GRIT8 = [median(AUCS(ii+150,2))/AUPR_avg(4), median(AUCS(ii+100,2))/AUPR_avg(3), median(AUCS(ii+200,2))/AUPR_avg(5), median(AUCS(ii,2))/AUPR_avg(1), median(AUCS(ii+50,2))/AUPR_avg(2), median(AUCS(ii+250,2))/AUPR_avg(6)];

load('../BEELINEbenchmark/Results/synthetic_results_15timepoints.mat')
GRIT15 = [median(AUCS(ii+150,2))/AUPR_avg(4), median(AUCS(ii+100,2))/AUPR_avg(3), median(AUCS(ii+200,2))/AUPR_avg(5), median(AUCS(ii,2))/AUPR_avg(1), median(AUCS(ii+50,2))/AUPR_avg(2), median(AUCS(ii+250,2))/AUPR_avg(6)];

scores = [GRIT8; GRIT15; scores];
scoreLimits = 1./AUPR_avg([4 3 5 1 2 6]);

xlims = [5 4.2 15 3.8 5.5 3.1];
figure('Position',[100 200 1200 640])
for jcase = 1:6
    subplot(2,3,jcase)
    barh(flipud(scores(:,jcase)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    hold on
    barh(flipud([scores(1:2,jcase); zeros(12,1)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
    plot(scoreLimits(jcase)*[1 1],[0.4 14.7],'LineWidth',1,'Color',[.7 0 .9])
    axis([0 xlims(jcase) 0.4 14.7])
    yticklabels(flipud(methodNames))
    set(gca,'FontSize',14)
    xlabel('AUPR-ratio','FontSize',16)
    title(datasetNames{jcase},'FontSize',18)
    grid
end


%% A summary plot

ranks = zeros(size(scores));
for jcase = 1:6
    [~,isort] = sort(scores(:,jcase),'descend');
    ranks(isort,jcase) = 1:size(ranks,1);
end

meanRank = mean(ranks,2);
rankRange = zeros(size(ranks,1),2);
for jmet = 1:size(ranks,1)
    rs = sort(ranks(jmet,:),'ascend');
    rankRange(jmet,:) = [mean(rs(1:3)) mean(rs(4:6))];
end
rankRange = size(ranks,1)-rankRange;

erb = [0 0 0 1 1 1; -.2 .2 0 0 .2 -.2];

indord = [2 1 3 7 6 4 5 10 9 8 12 11 14 13];

figure; hold on;
barh(flipud(size(ranks,1)-meanRank(indord)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
barh(flipud(size(ranks,1)-[meanRank([2 1]); 14*ones(12,1)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
for jmet = 1:size(ranks,1)
    plot(rankRange(indord(jmet),1) + (rankRange(indord(jmet),2)-rankRange(indord(jmet),1))*erb(1,:),15-jmet+erb(2,:),'k','LineWidth',1)
end
ax = gca;
ax.XGrid = 'on';
plot([0 0],[0 15],'k','LineWidth',.75)
plot([0 14],14.5999*[1 1],'Color',[.85 .85 .85])
axis([0 13.00001 .4 14.6])
yticks(1:14)
yticklabels(flipud(methodNames(indord)))
xticks(0:13)
xticklabels(14:-1:1)
set(gca,'FontSize',16)
xlabel('Rank','FontSize',20)
title('Synthetic','FontSize',22)

%% Supplementary figure: Effect of number of cells

GRIT15 = [];
for jc = 1:5
    ii = (jc:5:50);
    GRIT15 = [GRIT15; median(AUCS(ii+150,2))/AUPR_avg(4), median(AUCS(ii+100,2))/AUPR_avg(3), median(AUCS(ii+200,2))/AUPR_avg(5), median(AUCS(ii,2))/AUPR_avg(1), median(AUCS(ii+50,2))/AUPR_avg(2), median(AUCS(ii+250,2))/AUPR_avg(6)];
end

figure('Position',[100 200 1200 640])
for jcase = 1:6
    subplot(2,3,jcase)
    barh(flipud(GRIT15(:,jcase)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    ylim([0.4 5.7])
    yticklabels({'5000','2000','500','200','100'})
    set(gca,'FontSize',14)
    xlabel('AUPR-ratio','FontSize',16)
    ylabel('Number of cells','FontSize',16)
    title(datasetNames{jcase},'FontSize',18)
    grid
end



%% 


load('../BEELINEbenchmark/Results/synthetic_results_8timepoints.mat')



Ncells = [100 200 500 2000 5000];
ct = reshape(caseInfo(:,4),[50,6]);
ct = ct(:,[3 4 5]);

dims = [6, 7, 18];
%'CY','LI','LL'

ctCY = zeros(3,5);
ctLI = zeros(3,5);
ctLL = zeros(3,5);

for jnc = 1:5
    ctaux = sort(ct((1:5:50)+(jnc-1),1),'descend');
    ctCY(:,jnc) = [(ctaux(9)+ctaux(10))/2; median(ctaux); (ctaux(2)+ctaux(1))/2];
    
    ctaux = sort(ct((1:5:50)+(jnc-1),2),'descend');
    ctLI(:,jnc) = [(ctaux(9)+ctaux(10))/2; median(ctaux); (ctaux(2)+ctaux(1))/2];
    
    ctaux = sort(ct((1:5:50)+(jnc-1),3),'descend');
    ctLL(:,jnc) = [(ctaux(9)+ctaux(10))/2; median(ctaux); (ctaux(2)+ctaux(1))/2];
end
    
erb = [0 0 0 1 1 1; -.05 .05 0 0 .05 -.05];
cols = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840; 1 0 1; 0 1 1];


figure; hold on; grid on;
plot(log(Ncells),log(ctCY(2,:))/log(10),'o-','LineWidth',1.5)
plot(log(Ncells),log(ctLI(2,:))/log(10),'s--','LineWidth',1.5)
plot(log(Ncells),log(ctLL(2,:))/log(10),'d:','LineWidth',1.5)

for jnc = 1:5
    plot(log(Ncells(jnc))+erb(2,:),log(ctCY(1,jnc))/log(10) + (log(ctCY(3,jnc))/log(10)-log(ctCY(1,jnc))/log(10))*erb(1,:),'Color',cols(1,:))
    plot(log(Ncells(jnc))+erb(2,:),log(ctLI(1,jnc))/log(10) + (log(ctLI(3,jnc))/log(10)-log(ctLI(1,jnc))/log(10))*erb(1,:),'Color',cols(2,:))
    plot(log(Ncells(jnc))+erb(2,:),log(ctLL(1,jnc))/log(10) + (log(ctLL(3,jnc))/log(10)-log(ctLL(1,jnc))/log(10))*erb(1,:),'Color',cols(2,:))
end
xticks(log(Ncells))
xticklabels(Ncells)
set(gca,'FontSize',18)
xlabel('Total number of cells','FontSize',22)
ylabel('Computation time [s]','FontSize',22)
legend({'dim = 6 (CY)','dim = 7 (LI)','dim = 18 (LL)'},'FontSize',18,'Location','Northwest')
yticks([-1 0 1])
yticklabels([0.1 1 10])
axis([log(100)-.1 log(5000)+.1 -1.15 1.05])
