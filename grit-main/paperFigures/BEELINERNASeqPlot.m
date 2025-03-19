%RNASeq-data


%First section plots the figure in the supplementary material. The second
%section plots the summary figure in the main text.


%Results for other methods are from the source data file that is
%downloadable from the BEELINE publication website.
opts = detectImportOptions('../../BEELINE/41592_2019_690_MOESM14_ESM.csv');
TT = readtable('../../BEELINE/41592_2019_690_MOESM14_ESM.csv',opts);
methodNames = {'GRIT','PIDC','GENIE3','GRNBOOST2','SCODE','PPCOR','SINCERITIES'};
scores = [table2array(TT(:,6:11)) table2array(TT(:,15:20))];
caseNames = {'mHSC-E','mHSC-L','mHSC-GM','mESC','mDC','hESC','hHep'};

load('../BEELINEbenchmark/Results/RNASeq_AUCS_log.mat')

GRITscores = [AUCS{5,1}(1,3), AUCS{5,2}(1,3);
    AUCS{7,1}(1,3), AUCS{7,2}(1,3);
    AUCS{6,1}(1,3), AUCS{6,2}(1,3);
    AUCS{4,1}(1,3), AUCS{4,2}(1,3);
    AUCS{3,1}(1,3), AUCS{3,2}(1,3);
    AUCS{5,1}(2,3), AUCS{5,2}(2,3);
    AUCS{7,1}(2,3), AUCS{7,2}(2,3);
    AUCS{6,1}(2,3), AUCS{6,2}(2,3);
    AUCS{4,1}(2,3), AUCS{4,2}(2,3);
    AUCS{3,1}(2,3), AUCS{3,2}(2,3);
    AUCS{5,1}(3,3), AUCS{5,2}(3,3);
    AUCS{7,1}(3,3), AUCS{7,2}(3,3);
    AUCS{6,1}(3,3), AUCS{6,2}(3,3);
    AUCS{4,1}(3,3), AUCS{4,2}(3,3);
    AUCS{3,1}(3,3), AUCS{3,2}(3,3);
    AUCS{4,1}(4,3), AUCS{4,2}(4,3);
    AUCS{1,1}(1,3), AUCS{1,2}(1,3);
    AUCS{2,1}(1,3), AUCS{2,2}(1,3);
    AUCS{1,1}(2,3), AUCS{1,2}(2,3);
    AUCS{2,1}(2,3), AUCS{2,2}(2,3);
    AUCS{1,1}(3,3), AUCS{1,2}(3,3);
    AUCS{2,1}(3,3), AUCS{2,2}(3,3)];


scores = [GRITscores(:,1) scores(:,1:6) GRITscores(:,2) scores(:,7:12)];

caselines(1,:) = [1 2 3 4 5 17 18];
caselines(2,:) = [6 7 8 9 10 19 20];



figure('Position',[100 200 1000 840])
iaux = 1;
for jcase = 1:7

    %DIM 500, STRING
    subplot(7,4,1+4*(jcase-1))
    barh(fliplr(scores(caselines(1,jcase),1:7)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    hold on
    barh(fliplr([scores(caselines(1,jcase),1) zeros(1,6)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
    ylim([0.4 7.7])
    yticklabels(fliplr(methodNames))
    set(gca,'FontSize',12)
    grid
    if jcase == 7
        xlabel('EPR','FontSize',16)
    end
    ylabel(caseNames{jcase},'FontSize',16)
    
    
    
    %DIM 500, Chip
    subplot(7,4,2+4*(jcase-1))
    barh(fliplr(scores(caselines(2,jcase),1:7)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    hold on
    barh(fliplr([scores(caselines(2,jcase),1) zeros(1,6)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
    ylim([0.4 7.7])
    yticklabels({})
    set(gca,'FontSize',12)
    grid
    if jcase == 7
        xlabel('EPR','FontSize',16)
    end
    
    
    
     %DIM 1000, STRING
    subplot(7,4,3+4*(jcase-1))
    barh(fliplr(scores(caselines(1,jcase),8:14)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    hold on
    barh(fliplr([scores(caselines(1,jcase),8) zeros(1,6)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
    ylim([0.4 7.7])
    yticklabels({})
    set(gca,'FontSize',12)
    grid
    if jcase == 7
        xlabel('EPR','FontSize',16)
    end
    
    
    
    %DIM 1000, Chip
    subplot(7,4,4+4*(jcase-1))
    barh(fliplr(scores(caselines(2,jcase),8:14)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
    hold on
    barh(fliplr([scores(caselines(2,jcase),8) zeros(1,6)]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
    ylim([0.4 7.7]) 
    yticklabels({})
    set(gca,'FontSize',12) 
    grid
    if jcase == 7
        xlabel('EPR','FontSize',16)
    end
    
end
annotation('textbox', [.23, 1, 1, 0], 'EdgeColor',[1 1 1], 'string', 'TFs + 500 genes','FontSize',20)
annotation('textbox', [.64, 1, 1, 0], 'EdgeColor',[1 1 1], 'string', 'TFs + 1000 genes','FontSize',20)

annotation('textbox', [.171, .96, 0, 0], 'string', 'STRING','FontSize',18)
annotation('textbox', [.318, .96, 1, 0], 'EdgeColor',[1 1 1], 'string', 'Non-specific ChIP-Seq','FontSize',18)

annotation('textbox', [.585, .96, 0, 0], 'string', 'STRING','FontSize',18)
annotation('textbox', [.73, .96, 1, 0], 'EdgeColor',[1 1 1], 'string', 'Non-specific ChIP-Seq','FontSize',18)

%Reshape scores for the summary figure
scores = [scores([caselines(1,:), caselines(2,:)],1:7); scores([caselines(1,:), caselines(2,:)],8:14)]';


%% Summary figure

ranks = zeros(size(scores));
for jcase = 1:28
    [~,isort] = sort(scores(:,jcase),'descend');
    ranks(isort,jcase) = 1:size(ranks,1);
end

meanRank = mean(ranks,2);
rankRange = zeros(size(ranks,1),2);
for jmet = 1:size(ranks,1)
    rs = sort(ranks(jmet,:),'ascend');
    rankRange(jmet,:) = [mean(rs(1:14)) mean(rs(15:28))];
end
rankRange = size(ranks,1)-rankRange;

erb = [0 0 0 1 1 1; -.2 .2 0 0 .2 -.2];

indord = [2 3 4 1 5 7 6];
figure; hold on;
barh(flipud(size(ranks,1)-meanRank(indord)),'FaceColor',[176, 224, 230]/255,'EdgeColor',[176, 224, 230]/255)
barh(flipud(size(ranks,1)-[7;7;7;meanRank(1); 7;7;7]),'FaceColor',[233, 116, 81]/255,'EdgeColor',[233, 116, 81]/255)
for jmet = 1:size(ranks,1)
    plot(rankRange(indord(jmet),1) + (rankRange(indord(jmet),2)-rankRange(indord(jmet),1))*erb(1,:),8-jmet+erb(2,:),'k','LineWidth',1)
end
ax = gca;
ax.XGrid = 'on';
plot([0 0],[0 8],'k')
plot([0 7],7.5999*[1 1],'Color',[.85 .85 .85])
axis([0 6.00001 .4 7.6])
yticks(1:8)
yticklabels(fliplr(methodNames(indord)))
xticks(0:6)
xticklabels(7:-1:1)
set(gca,'FontSize',16)
xlabel('Rank','FontSize',20)
title('RNA-Seq','FontSize',22)










