

%Plot results of BEELINE RNASeq data with either no data transformation,
%square-root transformation or log-transformation. The plot is shown as a
%supplementary figure.


EPRtable500 = zeros(7,3);
EPRtable1000 = zeros(7,3);

load('../BEELINEbenchmark/Results/RNASeq_AUCS_noTransform.mat')
EPRtable500(1,1) = AUCS{5,1}(1,3);
EPRtable1000(1,1) = AUCS{5,2}(1,3);
EPRtable500(2,1) = AUCS{7,1}(1,3);
EPRtable1000(2,1) = AUCS{7,2}(1,3);
EPRtable500(3,1) = AUCS{6,1}(1,3);
EPRtable1000(3,1) = AUCS{6,2}(1,3);
EPRtable500(4,1) = AUCS{4,1}(1,3);
EPRtable1000(4,1) = AUCS{4,2}(1,3);
EPRtable500(5,1) = AUCS{3,1}(1,3);
EPRtable1000(5,1) = AUCS{3,2}(1,3);
EPRtable500(6,1) = AUCS{1,1}(1,3);
EPRtable1000(6,1) = AUCS{1,2}(1,3);
EPRtable500(7,1) = AUCS{2,1}(1,3);
EPRtable1000(7,1) = AUCS{2,2}(1,3);


load('../BEELINEbenchmark/Results/RNASeq_AUCS_sqrt.mat')
EPRtable500(1,2) = AUCS{5,1}(1,3);
EPRtable1000(1,2) = AUCS{5,2}(1,3);
EPRtable500(2,2) = AUCS{7,1}(1,3);
EPRtable1000(2,2) = AUCS{7,2}(1,3);
EPRtable500(3,2) = AUCS{6,1}(1,3);
EPRtable1000(3,2) = AUCS{6,2}(1,3);
EPRtable500(4,2) = AUCS{4,1}(1,3);
EPRtable1000(4,2) = AUCS{4,2}(1,3);
EPRtable500(5,2) = AUCS{3,1}(1,3);
EPRtable1000(5,2) = AUCS{3,2}(1,3);
EPRtable500(6,2) = AUCS{1,1}(1,3);
EPRtable1000(6,2) = AUCS{1,2}(1,3);
EPRtable500(7,2) = AUCS{2,1}(1,3);
EPRtable1000(7,2) = AUCS{2,2}(1,3);

load('../BEELINEbenchmark/Results/RNASeq_AUCS_log.mat')
EPRtable500(1,3) = AUCS{5,1}(1,3);
EPRtable1000(1,3) = AUCS{5,2}(1,3);
EPRtable500(2,3) = AUCS{7,1}(1,3);
EPRtable1000(2,3) = AUCS{7,2}(1,3);
EPRtable500(3,3) = AUCS{6,1}(1,3);
EPRtable1000(3,3) = AUCS{6,2}(1,3);
EPRtable500(4,3) = AUCS{4,1}(1,3);
EPRtable1000(4,3) = AUCS{4,2}(1,3);
EPRtable500(5,3) = AUCS{3,1}(1,3);
EPRtable1000(5,3) = AUCS{3,2}(1,3);
EPRtable500(6,3) = AUCS{1,1}(1,3);
EPRtable1000(6,3) = AUCS{1,2}(1,3);
EPRtable500(7,3) = AUCS{2,1}(1,3);
EPRtable1000(7,3) = AUCS{2,2}(1,3);


figure('Position',[300 200 800 400])
bar(EPRtable500)
grid
xticklabels({'mHSC-E','mHSC-L','mHSC-GM','mESC','mDC','hESC','hHep'})
set(gca,'FontSize',17)
axis([0.4 7.6 0 7])
ylabel('EPR','FontSize',20)
legend({'No transformation','sqrt-transformation','log-transformation'},'Location','Northeast','FontSize',18)
title('500 genes + TFs','FontSize',20)


figure('Position',[300 200 800 400])
bar(EPRtable1000)
grid
xticklabels({'mHSC-E','mHSC-L','mHSC-GM','mESC','mDC','hESC','hHep'})
set(gca,'FontSize',17)
axis([0.4 7.6 0 7])
ylabel('EPR','FontSize',20)
legend({'No transformation','sqrt-transformation','log-transformation'},'Location','Northeast','FontSize',18)
title('1000 genes + TFs','FontSize',20)


