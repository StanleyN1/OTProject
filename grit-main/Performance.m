function [AUROC, AUPR, TPR, FPR, PREC, CONF, EPR] = Performance(GT,Confidence,ignoreDiagonal,showWarning)

%Calculate AUROC and AUPR based on sorting the confidence matrix. The sizes
%of input matrices can be anything. If the ground truth matrix is smaller
%than the confidence, the confidence matrix is truncated to match the
%ground truth.

% Inputs:
% GT: Ground truth matrix. All non-zero values are interpreted as true
% links
%
% Confidence: doesn't need to be positive, but absolute value is
% interpreted as the confidence value.
% 
% ignoreDiagonal: logical variable indicating whether diagonal values are
% to be ignored.
%
% Outputs:
% AUROC/AUPR
% TPR: true positive rate (=recall) with confidence levels specified in CONF
% FPR: false positive rate (as above)
% PREC: precision (as above)
% CONF: Confidence levels for the other outputs
% EPR: Early precision ratio as defined in the BEELINE paper

if ~exist('ignoreDiagonal')
    ignoreDiagonal = false;
end

if ~exist('showWarning')
    showWarning = true;
end


if size(GT,1) ~= size(Confidence,1) || size(GT,2) ~= size(Confidence,2)
    warning('Sizes of input matrices do not match. Truncating the confidence matrix to match the ground truth.')
    Confidence = Confidence(1:size(GT,1),1:size(GT,2));
end

%Ignore diagonal if requested and vectorise both matrices
if ignoreDiagonal
    if showWarning
        disp(['Ignoring the diagonal elements.'])
    end
    if size(GT,2) > size(GT,1)
        GT = GT';
        Confidence = Confidence';
    end
    n = size(Confidence,1);
    A1 = zeros(numel(Confidence) - size(Confidence,2),1);
    GT1 = A1;
    for jg = 1:size(Confidence,2)
        A1((jg-1)*(n-1)+1:jg*(n-1)) = abs(Confidence([1:jg-1 jg+1:n],jg));
        GT1((jg-1)*(n-1)+1:jg*(n-1)) = abs(GT([1:jg-1 jg+1:n],jg));
    end    
else
    A1 = abs(Confidence(:));
    GT1 = abs(GT(:));
end

%Sort elements by decreasing confidence
[A1, ind] = sort(A1,'descend');
GT1 = GT1(ind);

%Early precision ratio with k = sum(GT1)
EPR = (sum(GT1(1:sum(GT1)))/sum(GT1)) / (sum(GT1)/length(GT1));



%Calculate false positive rate, true positive rate (=recall), and precision
trueTotal = sum(GT1 > 0);
falseTotal = length(GT1) - trueTotal;

%Initialise outputs
FPR = zeros(1,length(A1)+2);
TPR = zeros(1,length(A1)+2);
PREC = zeros(1,length(A1)+2);   
PREC(1) = 1;
foundSoFar = 0;

%Go through links one by one
iaux = 1;
while iaux < length(GT1) + .5
    
    %Check how many links have equal confidence
    nEqual = 1;
    while (iaux + nEqual < length(GT1) + .5) && (A1(iaux + nEqual) == A1(iaux))
        nEqual = nEqual + 1;
    end
    
    %How many of them were true/false
    TT = sum(GT1(iaux:iaux+nEqual-1));
    FF = nEqual - TT;
    foundSoFar = foundSoFar + sum(GT1(iaux:iaux+nEqual-1) > 0);
    
    %Update
    TPR(iaux+1:iaux+nEqual) = TPR(iaux) + TT/trueTotal*ones(1,nEqual);
    FPR(iaux+1:iaux+nEqual) = FPR(iaux) + FF/falseTotal*ones(1,nEqual);
    PREC(iaux+1:iaux+nEqual) = foundSoFar/(iaux+nEqual-1)*ones(1,nEqual);

    iaux = iaux + nEqual;
end
    

%Extend by proper last values
FPR(end) = 1;
TPR(end) = 1;
PREC(end) = sum(GT1 > 0)/length(GT1);
CONF = A1;

%Calculate AUROC and AUPR by trapezoidal integration (accurate for
%piecewise linear functions)
AUROC = 0;
AUPR = 0;
for jx = 2:length(FPR)
    AUROC = AUROC + (FPR(jx)-FPR(jx-1))*(TPR(jx)+TPR(jx-1))/2;
    AUPR = AUPR + (TPR(jx)-TPR(jx-1))*(PREC(jx)+PREC(jx-1))/2;
end

