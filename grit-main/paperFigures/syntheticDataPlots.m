%% Plots related to the experiment of Section 3.1

ncs = [1000 2000 3000 5000 7500 10000];

load('../SyntheticDataValidation/Results/Experiment_1_results_3timepoints/Experiment_1_results_3timepoints.mat')
Aerror3 = Aerror;

load('../SyntheticDataValidation/Results/Experiment_1_results_6timepoints/Experiment_1_results_6timepoints.mat')
Aerror6 = Aerror;

load('../SyntheticDataValidation/Results/Experiment_1_results_12timepoints/Experiment_1_results_12timepoints.mat')
Aerror12 = Aerror;


%80th percentiles obtained by taking the averages of highest and second
%highest values, and lowest and second lowest.
Range3 = zeros(2,6);
Range6 = zeros(2,6);
Range12 = zeros(2,6);
for jnc = 1:6
    s = sort(Aerror3(jnc,:),'ascend');
    Range3(:,jnc) = log([(s(1)+s(2))/2; (s(4)+s(5))/2])/log(10);
    
    s = sort(Aerror6(jnc,:),'ascend');
    Range6(:,jnc) = log([(s(1)+s(2))/2; (s(4)+s(5))/2])/log(10);
    
    s = sort(Aerror12(jnc,:),'ascend');
    Range12(:,jnc) = log([(s(1)+s(2))/2; (s(4)+s(5))/2])/log(10);
end


eb = [-.01 .01 0 0 -.01 .01; 0 0 0 1 1 1];
cols = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840; 1 0 1; 0 1 1; .3 .6 .2];

figure; hold on;
plot(log(ncs)/log(10),log(sum(Aerror3,2)/5)/log(10),'-o','LineWidth',1.5,'Color',cols(1,:))
plot(log(ncs)/log(10),log(sum(Aerror6,2)/5)/log(10),'--o','LineWidth',1.5,'Color',cols(2,:))
plot(log(ncs)/log(10),log(sum(Aerror12,2)/5)/log(10),':o','LineWidth',1.5,'Color',cols(3,:))

%Draw sloped gridlines
for jnc = 1:6
    plot(log([ncs(jnc) ncs(jnc)])/log(10),[-1 5],'Color',[.83 .83 .83])
end
for j2 = 0:8
    plot([2.97 4.03],-[0 4.03-2.97]/2 + log(2^j2)/log(10),'Color',[.83 .83 .83])
end

for j2 = 0:8
    plot([2.97 4.03],-[0 4.03-2.97]/2 + log(2^j2)/log(10),'Color',[.83 .83 .83])
end


plot(log(ncs)/log(10),log(sum(Aerror3,2)/5)/log(10),'-o','LineWidth',1.5,'Color',cols(1,:))
plot(log(ncs)/log(10),log(sum(Aerror6,2)/5)/log(10),'--o','LineWidth',1.5,'Color',cols(2,:))
plot(log(ncs)/log(10),log(sum(Aerror12,2)/5)/log(10),':o','LineWidth',1.5,'Color',cols(3,:))
for jnc = 1:6
    plot(log(ncs(jnc))/log(10)+eb(1,:)-.01,Range3(1,jnc) + (Range3(2,jnc)-Range3(1,jnc))*eb(2,:),'Color',cols(1,:),'LineWidth',1)
    plot(log(ncs(jnc))/log(10)+eb(1,:),Range6(1,jnc) + (Range6(2,jnc)-Range6(1,jnc))*eb(2,:),'Color',cols(2,:),'LineWidth',1)
    plot(log(ncs(jnc))/log(10)+eb(1,:)+.01,Range12(1,jnc) + (Range12(2,jnc)-Range12(1,jnc))*eb(2,:),'Color',cols(3,:),'LineWidth',1)
end
axis([2.97 4.03 -.22 log(44)/log(10)])
xticks(log(ncs)/log(10))
xticklabels(ncs)
yticks(log([1 2 4 8 16 32])/log(10))
yticklabels([1 2 4 8 16 32])
box on


set(gca,'FontSize',18)
xlabel('Number of cells per timepoint','FontSize',22)
ylabel('Squared model error','FontSize',22)
legend({'3 timepoints','6 timepoints','12 timepoints'},'Location','Northeast','FontSize',18)




% Computation time plot for the Supp. Mat.

%80th percentiles obtained by taking the averages of highest and second
%highest values, and lowest and second lowest.
Range = zeros(2,6);
for jnc = 1:6
    s = sort(compTimes(jnc,:),'ascend');
    Range(:,jnc) = log([(s(1)+s(2))/2; (s(4)+s(5))/2])/log(10);
end


eb = [-.01 .01 0 0 -.01 .01; 0 0 0 1 1 1];
cols = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840; 1 0 1; 0 1 1; .3 .6 .2];

figure; hold on; grid on;
plot(log(ncs)/log(10),log(sum(compTimes,2)/5)/log(10),'-o','LineWidth',1.5)


xx = log(ncs(2:end))/log(10);
xx = [xx; ones(size(xx))];
yy = log(sum(compTimes(2:end,:),2)/5)/log(10);

a = (xx*xx')^-1*xx*yy;
plot([3 4],a(1)*[3 4]+a(2),'--','LineWidth',1.5)


%Error bars, but they are so tiny that they are omitted
% for jnc = 1:6
%     plot(log(ncs(jnc))/log(10)+eb(1,:)-.01,Range(1,jnc) + (Range(2,jnc)-Range(1,jnc))*eb(2,:),'Color',cols(1,:),'LineWidth',1)
% end

axis([2.999 4 1.5 4])
xticks(log(ncs)/log(10))
xticklabels(ncs)
yticks(log([100 1000 10000])/log(10))
yticklabels([100 1000 10000])
set(gca,'FontSize',18)
xlabel('Number of cells per timepoint','FontSize',22)
ylabel('Computation time [s]','FontSize',22)




%% Plots related to the experiment of Section 3.2

load('../syntheticDataValidation/Results/Experiment_2_results.mat')

%GRIT results (for main text figure)
figure('Position',[400 300 700,320])
subplot(1,2,1)
boxplot(AUCS_GRN([1 3 5 7],:)')
set(gca,'FontSize',20)
ylabel('AUROC','FontSize',22)
xticklabels({'LD','LC','NL','Pr'})
grid
subplot(1,2,2)
boxplot(AUCS_GRN([2 4 6 8],:)')
set(gca,'FontSize',20)
ylabel('AUPR','FontSize',22)
xticklabels({'LD','LC','NL','Pr'})
grid

%Results with "bulk" data (for Supp. Mat.)
figure('Position',[400 300 700,320])
subplot(1,2,1)
boxplot(AUCS_Bulk([1 3 5 7],:)')
set(gca,'FontSize',20)
ylabel('AUROC','FontSize',22)
xticklabels({'LD','LC','NL','Pr'})
grid
subplot(1,2,2)
boxplot(AUCS_Bulk([2 4 6 8],:)')
set(gca,'FontSize',20)
ylabel('AUPR','FontSize',22)
xticklabels({'LD','LC','NL','Pr'})
grid

%Results from the A-matrix (for Supp. Mat.)
figure('Position',[400 300 700,320])
subplot(1,2,1)
boxplot(AUCS_A([1 3 5 7],:)')
set(gca,'FontSize',20)
ylabel('AUROC','FontSize',22)
xticklabels({'LD','LC','NL','Pr'})
grid
subplot(1,2,2)
boxplot(AUCS_A([2 4 6 8],:)')
set(gca,'FontSize',20)
ylabel('AUPR','FontSize',22)
xticklabels({'LD','LC','NL','Pr'})
grid


%% Plots related to the experiment of Section 3.3

load('../SyntheticDataValidation/Results/Experiment3_results.mat')


[AUROC(1), AUPR(1), TPR1, FPR1, PREC1, CONF1, EPR1] = Performance(GTpert1,pertRes1,false,false);
[AUROC(2), AUPR(2), TPR2, FPR2, PREC2, CONF2, EPR2] = Performance(GTpert2,pertRes2,false,false);
[AUROC(3), AUPR(3), TPR3, FPR3, PREC3, CONF3, EPR3] = Performance(GTpert3,pertRes3,false,false);
[AUROC(4), AUPR(4), TPR4, FPR4, PREC4, CONF4, EPR4] = Performance(GTpert4,pertRes4,false,false);
[AUROC(5), AUPR(5), TPR5, FPR5, PREC5, CONF5, EPR5] = Performance(GTpert5,pertRes5,false,false);
[AUROC(6), AUPR(6), TPR6, FPR6, PREC6, CONF6, EPR6] = Performance(GTpert6,pertRes6,false,false);



[AUROC(7), AUPR(7), TPR7, FPR7, PREC7, CONF7, EPR7] = Performance(GTpert1,corRes1,false,false);
[AUROC(8), AUPR(8), TPR8, FPR8, PREC8, CONF8, EPR8] = Performance(GTpert2,corRes2,false,false);
[AUROC(9), AUPR(9), TPR9, FPR9, PREC9, CONF9, EPR9] = Performance(GTpert3,corRes3,false,false);
[AUROC(10), AUPR(10), TPR10, FPR10, PREC10, CONF10, EPR10] = Performance(GTpert4,corRes4,false,false);
[AUROC(11), AUPR(11), TPR11, FPR11, PREC11, CONF11, EPR11] = Performance(GTpert5,corRes5,false,false);
[AUROC(12), AUPR(12), TPR12, FPR12, PREC12, CONF12, EPR12] = Performance(GTpert6,corRes6,false,false);


figure; hold on; grid on;
syms = {'s','o','^','d','v','*'};
for jexp = 1:6
    plot(AUROC(jexp),AUPR(jexp),syms{jexp},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',12)
end
for jexp = 1:6
    plot(AUROC(jexp+6),AUPR(jexp+6),syms{jexp},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',12)
end
%axis([0.33 1 0 1])
set(gca,'FontSize',18)
xlabel('AUROC','FontSize',22)
ylabel('AUPR','FontSize',22)
%legend({'60% increase for one gene','60% increase for two genes','60% increase for three genes','30% increase for one gene','30% increase for two genes','30% increase for three genes'},'FontSize',16,'Location','Northwest')
%legend({'60% increase for one gene','60% increase for two genes','60% increase for three genes','30% increase for one gene','30% increase for two genes','30% increase for three genes'},'FontSize',17,'Position',[.275 .725 .2 .2])


%For the separate legend
figure('Position',[500 360 100 260]); hold on;
plot(0,4,syms{1},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,3,syms{2},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,2,syms{3},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,1,syms{4},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,0,syms{5},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,-1,syms{6},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)

plot(1,4,syms{1},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,3,syms{2},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,2,syms{3},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,1,syms{4},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,0,syms{5},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,-1,syms{6},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
axis([-.2 1.2 -1.01 4.01])
axis off


%% Plots related to the experiment of Section 3.4


load('../SyntheticDataValidation/Results/Experiment4_results.mat')


[AUROC(1), AUPR(1), TPR1, FPR1, PREC1, CONF1, EPR1] = Performance(GTmut1,pertRes1,false,false);
[AUROC(2), AUPR(2), TPR2, FPR2, PREC2, CONF2, EPR2] = Performance(GTmut1,pertRes2,false,false);
[AUROC(3), AUPR(3), TPR3, FPR3, PREC3, CONF3, EPR3] = Performance(GTmut2,pertRes3,false,false);
[AUROC(4), AUPR(4), TPR4, FPR4, PREC4, CONF4, EPR4] = Performance(GTmut2,pertRes4,false,false);

[AUROC(5), AUPR(5), TPR5, FPR5, PREC5, CONF5, EPR5] = Performance(GTmut1,corRes1,false,false);
[AUROC(6), AUPR(6), TPR6, FPR6, PREC6, CONF6, EPR6] = Performance(GTmut1,corRes2,false,false);
[AUROC(7), AUPR(7), TPR7, FPR7, PREC7, CONF7, EPR7] = Performance(GTmut2,corRes3,false,false);
[AUROC(8), AUPR(8), TPR8, FPR8, PREC8, CONF8, EPR8] = Performance(GTmut2,corRes4,false,false);

figure; hold on; grid on;
syms = {'s','o','^','d'};
for jexp = [1 3 2 4]
    plot(AUROC(jexp),AUPR(jexp),syms{jexp},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',12)
end
for jexp = [1 3 2 4]
    plot(AUROC(jexp+4),AUPR(jexp+4),syms{jexp},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',12)
end
set(gca,'FontSize',18)
xlabel('AUROC','FontSize',22)
ylabel('AUPR','FontSize',22)
%legend({'50% loss for one regulation','50% loss for two regulations','20% loss for one regulation','20% loss for two regulations'},'FontSize',18,'Position',[.28 .73 .2 .2])


%For the separate legend
figure('Position',[500 360 100 180]); hold on;
plot(0,4,syms{1},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,3,syms{3},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,2,syms{2},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)
plot(0,1,syms{4},'Color',[.1 .1 .7],'MarkerFaceColor',[.1 .1 .7],'MarkerSize',20)

plot(1,4,syms{1},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,3,syms{3},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,2,syms{2},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
plot(1,1,syms{4},'Color',[1 .65 .65],'MarkerFaceColor',[1 .65 .65],'MarkerSize',20)
axis([-.2 1.2 .99 4.01])
axis off
