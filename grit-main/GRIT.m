function [GRN,A,transportMap,J,difs,out] = GRIT(scdata,Tgrid,TFflag,branchId,opts)


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

%GRN inference step
perturbations = struct;
perturbations.npert = 0;
perturbations.mutGenes = [];
GRN = GRITgrn(XX,YY,D,WW,corNet,indw,TFflag,perturbations,opts);








