%Curated data

opts = detectImportOptions('../../BEELINE/41592_2019_690_MOESM13_ESM.csv');
TT = readtable('../../BEELINE/41592_2019_690_MOESM13_ESM.csv',opts);
methodNames = TT.Var1(1:end);
datasetNames = {'mCAD','mCAD','VSC','VSC','HSC','HSC','GSD','GSD'};
scores = table2array(TT(:,2:9));
methodNames = ['GRIT, 8'; 'GRIT, 15'; methodNames];




load('../BEELINEbenchmark/Results/Curated_random_AUPR.mat')

load('../BEELINEbenchmark/Results/curated_results_8timepoints.mat')
GRIT8 = [median(AUCS(61:3:90,2))/AUPR_avg(3), median(AUCS(91:3:120,2))/AUPR_avg(4), median(AUCS(31:3:60,2))/AUPR_avg(2), median(AUCS(1:3:30,2))/AUPR_avg(1)];
GRIT8 = [GRIT8, median(AUCS(61:3:90,3)), median(AUCS(91:3:120,3)), median(AUCS(31:3:60,3)), median(AUCS(1:3:30,3))];

load('../BEELINEbenchmark/Results/curated_results_15timepoints.mat')
GRIT15 = [median(AUCS(61:3:90,2))/AUPR_avg(3), median(AUCS(91:3:120,2))/AUPR_avg(4), median(AUCS(31:3:60,2))/AUPR_avg(2), median(AUCS(1:3:30,2))/AUPR_avg(1)];
GRIT15 = [GRIT15, median(AUCS(61:3:90,3)), median(AUCS(91:3:120,3)), median(AUCS(31:3:60,3)), median(AUCS(1:3:30,3))];

scores = [GRIT8; GRIT15; scores];


figure('Position',[100 200 1300 640])
caseinds = [1 5 2 6 3 7 4 8];
iipl = [1 2 3; 4 5 6; 8 9 10; 11 12 13; 14 15 16; 17 18 19; 21 22 23; 24 25 26];
for jcase = 1:8
    subplot(2,13,iipl(jcase,:))
    barh(flipud(scores(:,caseinds(jcase))),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    hold on
    barh(flipud([scores(1:2,caseinds(jcase)); zeros(12,1)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
    ylim([0.4 14.7])
    set(gca,'FontSize',14)
    if mod(jcase,2) == 1
        xlabel('AUPR-ratio','FontSize',16)
        yticklabels(flipud(methodNames))
        title([repmat(' ',1,38) datasetNames{jcase}],'FontSize',22)
    else
        xlabel('EPR','FontSize',16)
        yticklabels({})
    end
    grid
end


%% A summary plot

ranks = zeros(size(scores));
for jcase = 1:8
    [~,isort] = sort(scores(:,jcase),'descend');
    ranks(isort,jcase) = 1:size(ranks,1);
end

meanRank = mean(ranks,2);
rankRange = zeros(size(ranks,1),2);
for jmet = 1:size(ranks,1)
    rs = sort(ranks(jmet,:),'ascend');
    rankRange(jmet,:) = [mean(rs(1:4)) mean(rs(5:8))];
end
rankRange = size(ranks,1)-rankRange;

erb = [0 0 0 1 1 1; -.2 .2 0 0 .2 -.2];


indord = [1 4 2 3 7 5 6 8 10 9 14 12 11 13];

figure; hold on;
barh(flipud(size(ranks,1)-meanRank(indord)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
barh(flipud(size(ranks,1)-[meanRank(1); 14; meanRank(2); 14*ones(11,1)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
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
title('Curated','FontSize',22)


%% Supplementary figure: Effect of dropouts

GRIT15 = [];
for jc = 0:2
    GRITaux = [median(AUCS((61:3:90)+jc,2))/AUPR_avg(3), median(AUCS((91:3:120)+jc,2))/AUPR_avg(4), median(AUCS((31:3:60)+jc,2))/AUPR_avg(2), median(AUCS((1:3:30)+jc,2))/AUPR_avg(1)];
    GRITaux = [GRITaux, median(AUCS((61:3:90)+jc,3)), median(AUCS((91:3:120)+jc,3)), median(AUCS((31:3:60)+jc,3)), median(AUCS((1:3:30)+jc,3))];
    GRIT15 = [GRIT15; GRITaux];
end

figure('Position',[100 200 1300 450])
caseinds = [1 5 2 6 3 7 4 8];
iipl = [1 2 3; 4 5 6; 8 9 10; 11 12 13; 14 15 16; 17 18 19; 21 22 23; 24 25 26];
for jcase = 1:8
    subplot(2,13,iipl(jcase,:))
    barh(flipud(GRIT15(:,caseinds(jcase))),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    ylim([0.4 3.7])
    set(gca,'FontSize',14)
    if mod(jcase,2) == 1
        xlabel('AUPR-ratio','FontSize',16)
        yticklabels({'70%','50%','0%'})
        ylabel('Share of missing data','FontSize',16)
        title([repmat(' ',1,38) datasetNames{jcase}],'FontSize',22)
    else
        xlabel('EPR','FontSize',16)
        yticklabels({})
    end
    grid
end

