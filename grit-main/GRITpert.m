function [GRN,A,transportMap,J,difs,out] = GRITpert(scdata,Tgrid,TFflag,branchId,perturbations,opts)

 %   Copyright 2024 Atte AALTO, FRANCOIS LAMOLINE, JORGE GONCALVES, 
 %   JOHAN KARLSSON, ISABEL HAASLER
 %
 %   Licensed under the Apache License, Version 2.0 (the "License");
 %   you may not use this file except in compliance with the License.
 %   You may obtain a copy of the License at
 % 
 %       http://www.apache.org/licenses/LICENSE-2.0
 %
 %   Unless required by applicable law or agreed to in writing, software
 %   distributed under the License is distributed on an "AS IS" BASIS,
 %   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 %   See the License for the specific language governing permissions and
 %   limitations under the License.

if ~exist('opts')
    opts = struct;
end

%Model selection iteration
[XX,YY,transportMap,J,A,D,WW,corNet,indw,indexp,TFflag,difs,out,opts] = GRITmodelSelect(scdata,Tgrid,TFflag,branchId,opts);


%Handle perturbations
perturbations.npert = 0;
if isfield(perturbations,'Perturbation')
    if length(perturbations.Perturbation) ~= max(indexp)
        error('Perturbation data size inconsistent')
    end
    
%     perts = zeros(1,max(indexp));
%     for jexp = 1:length(perturbations.Perturbation)
%         perts(perturbations.Perturbation{jexp},jexp) = 1;
%     end
    
    perts = [];
    for jexp = 1:length(perturbations.Perturbation)
        perts = [perts perturbations.Perturbation{jexp}(:)'];
    end
    perts = unique(perts);
    perturbations.npert = length(perts);
    
    Xaug = [];
    for jp = perts
        Xgene = [];
        for jexp = 1:length(perturbations.Perturbation)
            if ismember(jp,perturbations.Perturbation{jexp})
                Xgene = [Xgene, sum(XX(size(scdata{1},1)+1:end,indexp == jexp),1)];
            else
                Xgene = [Xgene zeros(1,sum(indexp == jexp))];
            end
        end
        Xaug = [Xaug; Xgene];
    end
    Daug = mean(D(size(scdata{1},1)+1:end))*ones(length(perts),1);
    D = [D(1:size(scdata{1},1)); Daug; D(size(scdata{1},1)+1:end)];
    TFflag = [TFflag(1:size(scdata{1},1)); true(length(perts),1); TFflag(size(scdata{1},1)+1:end)];
    XX = [XX(1:size(scdata{1},1),:); Xaug; XX(size(scdata{1},1)+1:end,:)];
    
end

%Handle mutations
if isfield(perturbations,'Mutation')
    if length(perturbations.Mutation) ~= max(indexp)
        error('Mutation data size inconsistent')
    end
    
    mutGenes = [];
    for jexp = 1:length(perturbations.Mutation)
        mutGenes = [mutGenes perturbations.Mutation{jexp}(:)'];
    end
    mutGenes = unique(mutGenes);
    perturbations.mutGenes = mutGenes;
    out.mutGenes = mutGenes;
    
    Daug = [];
    TFaug = [];
    Xaug = [];
    for jg = mutGenes
        Xgene = [];
        for jexp = 1:length(perturbations.Mutation)
            if ismember(jg,perturbations.Mutation{jexp})
                Xgene = [Xgene XX(jg,indexp == jexp)];
            else
                Xgene = [Xgene zeros(1,sum(indexp == jexp))];
            end
        end
        
        Xaug = [Xaug; Xgene];
        Daug = [Daug; D(jg)];
        TFaug = [TFaug; TFflag(jg)];
        
    end
    D = [D(1:size(scdata{1},1)+perturbations.npert); Daug; D(size(scdata{1},1)+perturbations.npert+1:end)];
    TFflag = [TFflag(1:size(scdata{1},1)+perturbations.npert); TFaug; TFflag(size(scdata{1},1)+perturbations.npert+1:end)];
    XX = [XX(1:size(scdata{1},1)+perturbations.npert,:); Xaug; XX(size(scdata{1},1)+perturbations.npert+1:end,:)];
else
    perturbations.mutGenes = [];
end
   
%Make sure TFflag is logical
TFflag = TFflag > 0;
    
%GRN inference step
GRN = GRITgrn(XX,YY,D,WW,corNet,indw,TFflag,perturbations,opts);








