function cc = corChange(scdataC,scdataM,mutGene)

Y = [];
for jt = 1:length(scdataC)
    Y = [Y scdataC{jt}];
end

Y = Y-mean(Y,2);
P1 = Y*Y';
sc = diag(P1).^-.5;
P1 = sc.*P1.*sc';

Y = [];
for jt = 1:length(scdataM)
    Y = [Y scdataM{jt}];
end
Y = Y-mean(Y,2);
P2 = Y*Y';
sc = diag(P2).^-.5;
P2 = sc.*P2.*sc';

if mutGene > 0
    cc = P1(:,mutGene)-P2(:,mutGene,:);
else
    cc = sum(abs(P1-P2),2);
end



