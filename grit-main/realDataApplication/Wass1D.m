function D = Wass1D(X,Y)

%Wasserstein 2-distance between 1D point clouds X and Y, calculated as the
%L1 integral of the difference of the cumulative distributions.


%Reshape and attach indicator variable
NX = length(X);
NY = length(Y);
X = [X(:)'; ones(1,NX)];
Y = [Y(:)'; zeros(1,NY)];

%Concatenate and sort
X = [X,Y];
[~,sortInd] = sort(X(1,:),'ascend');
X = X(:,sortInd);

%Cumulative distribution values
CX = X(2,1)/NX;
CY = (1-X(2,1))/NY;

D = 0;
for jx = 2:size(X,2)
    D = D + (X(1,jx)-X(1,jx-1))*abs(CX-CY);
    
    CX = CX + X(2,jx)/NX;
    CY = (1-X(2,jx))/NY;
end























