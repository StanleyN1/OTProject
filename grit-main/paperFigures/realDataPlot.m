%Plot the histograms of confidence scores and indicate the top genes.



%% PINK1, perturbation

TP = readtable('../realDataApplication/Results/PINK1_dim1500_case4.csv');

close all
figure; grid on; hold on;
histogram(TP.Confidence,'FaceColor',[176, 224, 230]/255,'EdgeColor',.8*[176, 224, 230]/255)
plot([0 0],[0 600],'k')
set(gca,'FontSize',16)
xlabel('Score','FontSize',20)
ylabel('Number of genes','FontSize',20)
title('{\it PINK1} (Perturbation target inference)','FontSize',21)

text(.975,6,'PCSK1, AP5M1','FontSize',15,'Rotation',90,'FontAngle','italic')
text(.925,6,'MKI67, LIN7A','FontSize',15,'Rotation',90,'FontAngle','italic') 
text(.875,6,'EGFL6, DAB2, SYNE2','FontSize',15,'Rotation',90,'FontAngle','italic') 

fill([.825 .825 .99 .99 .825],[5 460 460 300 300],[.985 .985 .985],'LineWidth',1)
text(.835,440,'CENPF','FontSize',15,'FontAngle','italic')
text(.835,410,'EXOC5','FontSize',15,'FontAngle','italic')
text(.835,380,'CAMK2D','FontSize',15,'FontAngle','italic')
text(.835,350,'DDX5','FontSize',15,'FontAngle','italic')
text(.835,320,'SLITRK6','FontSize',15,'FontAngle','italic')


fill([.775 .775 .61 .61 .775],[6 460 460 270 270],[.985 .985 .985],'LineWidth',1)
text(.62,440,'ELAVL4','FontSize',15,'FontAngle','italic')
text(.62,410,'ATP2B1','FontSize',15,'FontAngle','italic')
text(.62,380,'SNHG5','FontSize',15,'FontAngle','italic')
text(.62,350,'CDH2','FontSize',15,'FontAngle','italic')
text(.62,320,'PCM1','FontSize',15,'FontAngle','italic')
text(.62,290,'APLP2','FontSize',15,'FontAngle','italic')


fill([.725 .725  .405 .405 .57 .57 .725],[15 90 90 550 550 90 90],[.985 .985 .985],'LineWidth',1)
text(.415,530,'ENAH','FontSize',15,'FontAngle','italic')
text(.415,500,'ASNS','FontSize',15,'FontAngle','italic')
text(.415,470,'RAD21','FontSize',15,'FontAngle','italic')
text(.415,440,'ZFHX4','FontSize',15,'FontAngle','italic')
text(.415,410,'VIM','FontSize',15,'FontAngle','italic')
text(.415,380,'HACD3','FontSize',15,'FontAngle','italic')
text(.415,350,'CHRNA5','FontSize',15,'FontAngle','italic')
text(.415,320,'CRABP1','FontSize',15,'FontAngle','italic')
text(.415,290,'SLK','FontSize',15,'FontAngle','italic')
text(.415,260,'EGLN3','FontSize',15,'FontAngle','italic')
text(.415,230,'GPC3','FontSize',15,'FontAngle','italic')
text(.415,200,'VCAN','FontSize',15,'FontAngle','italic')
text(.415,170,'PITX2','FontSize',15,'FontAngle','italic')
text(.415,140,'KIF5B','FontSize',15,'FontAngle','italic')
text(.415,110,'LGI1','FontSize',15,'FontAngle','italic')




%% LRRK2

TL = readtable('../realDataApplication/Results/LRRK2_dim1000.csv');
close all
figure; grid on; hold on;
histogram(TL.Confidence,'FaceColor',[176, 224, 230]/255,'EdgeColor',.8*[176, 224, 230]/255)
plot([0 0],[0 500],'k')
set(gca,'FontSize',16)
xlabel('Score','FontSize',20)
ylabel('Number of genes','FontSize',20)
title('LRRK2','FontSize',21,'FontAngle','italic')

fill([.975 .975 .795 .795 .975],[6 475 475 335 335],[.985 .985 .985],'LineWidth',1)
text(.805,460,'CLU','FontSize',15,'FontAngle','italic')
text(.805,438,'IGFBP7','FontSize',15,'FontAngle','italic')
text(.805,416,'NEAT1','FontSize',15,'FontAngle','italic')
text(.805,394,'GFAP','FontSize',15,'FontAngle','italic')
text(.805,372,'HLA-E','FontSize',15,'FontAngle','italic')
text(.805,350,'GNG11','FontSize',15,'FontAngle','italic')

fill([.925 .925 .745 .745 .925],[5 315 315 197 197],[.985 .985 .985],'LineWidth',1)
text(.755,300,'B2M','FontSize',15,'FontAngle','italic')
text(.755,278,'SCRG1','FontSize',15,'FontAngle','italic')
text(.755,256,'SPARCL1','FontSize',15,'FontAngle','italic')
text(.755,234,'SAT1','FontSize',15,'FontAngle','italic')
text(.755,212,'XIST','FontSize',15,'FontAngle','italic')

fill([.875 .875 .695 .695 .875],[2 177 177 125 125],[.985 .985 .985],'LineWidth',1)
text(.705,162,'ATP5E','FontSize',15,'FontAngle','italic')
text(.705,140,'HSP90B1','FontSize',15,'FontAngle','italic')
%text(.875,10,'ATP5E, HSP90B1','FontSize',15,'Rotation',90,'FontAngle','italic') 

fill([.825 .825 .67 .67 .49 .49 .67 .67 .825],[6 90 90 475 475 335 335 90 90],[.985 .985 .985],'LineWidth',1)
text(.5,460,'MT-RNR2','FontSize',15,'FontAngle','italic')
text(.5,438,'RAB31','FontSize',15,'FontAngle','italic')
text(.5,416,'MT-TP','FontSize',15,'FontAngle','italic')
text(.5,394,'TFPI2','FontSize',15,'FontAngle','italic')
text(.5,372,'RCAN1','FontSize',15,'FontAngle','italic')
text(.5,350,'ITM2C','FontSize',15,'FontAngle','italic')

fill([.775 .775 .645 .645 .445 .445 .645 .645 .775],[11 60 60 310 310 60 60 60 60],[.985 .985 .985],'LineWidth',1)
text(.455,295,'NR2F1','FontSize',15,'FontAngle','italic')
text(.455,273,'ITM2B','FontSize',15,'FontAngle','italic')
text(.455,251,'MT-ND4','FontSize',15,'FontAngle','italic')
text(.455,229,'MT-ND5','FontSize',15,'FontAngle','italic')
text(.455,207,'MT-TS2','FontSize',15,'FontAngle','italic')
text(.455,185,'TMSB4X','FontSize',15,'FontAngle','italic')
text(.455,163,'GPM6B','FontSize',15,'FontAngle','italic')
text(.455,141,'VGF','FontSize',15,'FontAngle','italic')
text(.455,119,'MTRNR2L1','FontSize',15,'FontAngle','italic')
text(.455,97,'RPL41','FontSize',15,'FontAngle','italic')
text(.455,75,'PEG10','FontSize',15,'FontAngle','italic')



%% PINK1 mutation, dim 1500

TP = readtable('../realDataApplication/Results/PINK1_dim1500_case2.csv');
close all
figure; grid on; hold on;

subplot(5,1,1)
histogram(TP.Confidence,0:.01:.2,'FaceColor',[176, 224, 230]/255,'EdgeColor',.8*[176, 224, 230]/255)
set(gca,'FontSize',16)
grid on;
hold on;
plot([0 0],[0 1200],'k')

title('{\it PINK1} (Mutation effect inference)','FontSize',21)
axis([0 .2 1100 1200])
xticklabels({})
yticks(1100:100:1200)

subplot(5,1,2:5)
histogram(TP.Confidence,0:.01:.2,'FaceColor',[176, 224, 230]/255,'EdgeColor',.8*[176, 224, 230]/255)
grid on;
hold on;
plot([0 0],[0 1200],'k')
axis([0 .2 0 200])
set(gca,'FontSize',16)
xlabel('Score','FontSize',20)
ylabel('Number of genes','FontSize',20)

text(.195,3,'COL1A2','FontSize',15,'Rotation',90,'FontAngle','italic') 
text(.135,3,'ZNF518A','FontSize',15,'Rotation',90,'FontAngle','italic') 
text(.125,3,'ZFYVE16','FontSize',15,'Rotation',90,'FontAngle','italic') 
text(.095,3,'GPC3','FontSize',15,'Rotation',90,'FontAngle','italic') 
text(.085,3,'ODF2L, AKAP13','FontSize',15,'Rotation',90,'FontAngle','italic') 
text(.075,5,'COL3A1, RGS2, ZNF91','FontSize',15,'Rotation',90,'FontAngle','italic') 

fill([.065 .065 .032 .032 .065],[6 154 154 80 80],[.985 .985 .985],'LineWidth',1)
text(.034,146,'CCT2','FontSize',15,'FontAngle','italic')
text(.034,134,'PLCL1','FontSize',15,'FontAngle','italic')
text(.034,122,'ARMCX3','FontSize',15,'FontAngle','italic')
text(.034,110,'NCALD','FontSize',15,'FontAngle','italic')
text(.034,98,'P4HA1','FontSize',15,'FontAngle','italic')
text(.034,86,'PPM1K','FontSize',15,'FontAngle','italic')



