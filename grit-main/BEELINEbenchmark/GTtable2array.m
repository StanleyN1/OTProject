function [GT, inputInds] = GTtable2array(GTtable,geneNames,removeDiagonal)

if ~exist('removeDiagonal')
    removeDiagonal = false;
end


%Sort table by input genes as one input often has several outputs
GTtable = sortrows(GTtable,'Gene1');
gAux = GTtable.Gene1{1};
jin = find(strcmp(GTtable.Gene1{1},geneNames));

GT = zeros(length(geneNames),length(geneNames));
for jlink = 1:size(GTtable,1)
    
    %Check if the input changed
    if ~strcmp(GTtable.Gene1{jlink},gAux)
        gAux = GTtable.Gene1{jlink};
        jin = find(strcmp(GTtable.Gene1{jlink},geneNames));
    end
    
    %Find output only if input is included in geneNames
    if ~isempty(jin)
        jout = find(strcmp(GTtable.Gene2{jlink},geneNames));
    
        if ~isempty(jout)
            GT(jout,jin) = 1;
        end
    end
end

%Set diagonal to zeros if requested
if removeDiagonal
    GT = GT.*(1-eye(size(GT,1)));
end
    

if nargout > 1
    inputInds = find(sum(GT) > 0);
    GT = GT(:,inputInds);
end
    
    










