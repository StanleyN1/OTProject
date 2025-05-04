%% Load and plot example data
load('syntheticDataValidation/exampleData.mat')

Y = [];
for jt = 1:length(scdata)
    Y = [Y scdata{jt}];
end
Y = Y - mean(Y,2);
[U,~,~] = svd(Y);
figure; hold on; grid on;
for jt = 1:length(scdata)
    plot(U(:,1)'*scdata{jt},U(:,2)'*scdata{jt},'x')
end
set(gca,'FontSize',12)
xlabel('PC 1','FontSize',14)
ylabel('PC 2','FontSize',14)



%% Run GRIT with default options and calculate performance metrics
opts = struct;
[GRN,A,transportMap] = GRIT(scdata,Tgrid,[],[],opts);

%Performance
[AUROC, AUPR] = Performance(GT,GRN,true)







