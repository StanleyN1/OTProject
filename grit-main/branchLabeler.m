function branchId = branchLabeler(M,clusterLabels,threshold)

%Inputs: 
%
%M: cell structure, where M{k} is the transport matrix between time points
%T_k and T_{k+1}
%
%clusterLabels: a m*n matrix where m is the number of clusters, and n is
%the number of cells measured in the last time point. The matrix entries
%are zeros and ones with exactly one one per column indicating to which
%cluster the corresponding cell belongs to.
%
%threshold: which percentage of mass is required to be associated to a
%branch, so that the cell will be assigned to that branch

%Check input consistency
if size(clusterLabels,2) ~= size(M{end},2)
    error('The cluster label size does not match the size of the last transport matrix')
end

%Set threshold default
if ~exist('threshold')
    threshold = .5/size(clusterLabels,1);
end
   
%Initialisation
branchId = cell(1,length(M)+1);
branchId{end} = clusterLabels;

T = eye(size(M{end},2));
for jt = 1:length(M)
    
    %Form the scaled transport matrix from time jt to final time point
    T = M{end-jt+1}*T;
    Tnorm = T./sum(T,2);
    
    %Find the percentage of descendants belonging to each branch
    branchShare = branchId{end}*Tnorm';
    branchId{end-jt} = branchShare > threshold;
    
    %Make sure each cell belongs to some branch
    if threshold > 1/size(clusterLabels,1)
        for jcell = 1:size(branchShare,2)
            [~,imax] = max(branchShare(:,jcell));
            branchId{end-jt}(imax,jcell) = 1;
        end
    end
          
    %Convert from logical to number
    branchId{end-jt} = double(branchId{end-jt}); 
end




