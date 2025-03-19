function GRN = GRITgrn(XX,YY,D,WW,corNet,indw,TFflag,perturbations,opts)

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

if ~isfield(opts,'forcePert')
    opts.forcePert = false;
end
if opts.par > 1
    parpool(opts.par);
end

%Number of genes
ndim = size(YY,1);

%Number of perturbations
npert = perturbations.npert;

%Number of Mutations
nmut = length(perturbations.mutGenes);

%Number of branches
nbr = size(XX,1)-ndim-nmut-npert;

%=== Network inference ===

maxReg = min(opts.maxReg,sum(TFflag(1:ndim)));
parfor (jg = 1:ndim, opts.par)
    
    %Initialise the regressor set
    if maxReg == sum(TFflag(1:ndim))
        TFloc = TFflag;
        fwd = false;
    else
        TFloc = TFflag;
        TFloc(jg) = true;
        [~,indsrt] = sort(TFloc(1:ndim)'.*corNet(jg,:).^2,'descend');
        TFloc = [false(ndim+nmut,1); true(nbr,1)];
        TFloc = [false(ndim,1); true(npert,1); false(nmut,1); true(nbr,1)];
        TFloc(indsrt(1:maxReg-5)) = true;
        fwd = true;
    end
    
    %Add target gene to regressor set
    TFloc(jg) = true;
    
    %Add mutations to regressor set
    if opts.forcePert
        TFloc(perturbations.mutGenes) = true;
    end
    for jmut = 1:length(perturbations.mutGenes)
        TFloc(ndim+npert+jmut) = TFloc(perturbations.mutGenes(jmut));
    end
    
    TFlocList = find(TFloc)';
    TFleft = and(TFflag(1:ndim),~TFloc(1:ndim));
    TFleftList = find(TFleft);
    TFleftList = TFleftList(:);
    TFlocList = TFlocList(:)';
    
    CF{jg} = zeros(2,ndim);
    CB{jg} = zeros(2,ndim+npert+nmut);
    
    while sum(TFloc) > nbr+1 || fwd      
        
        %Switch to backward greedy steps once the forward sweep is complete
        if sum(TFloc(1:ndim)) == maxReg
            fwd = false;
        end
        
        %Solve A (row) with regressors indicated by TFloc
        G1 = ((XX(TFloc,:).*WW(jg,indw))*XX(TFloc,:)' + diag(D(TFloc)))^-1;
        Atemp = (YY(jg,:).*WW(jg,indw))*XX(TFloc,:)'*G1;
        Jnew = sum(WW(jg,indw).*(YY(jg,:) - Atemp*XX(TFloc,:)).^2,2) + sum(Atemp.^2*diag(D(TFloc)),2);
        
        
        if fwd %Forward greedy step
        
            %Check regressor importance for candidate regressors
            DeltaJ = zeros(1,length(TFleftList));
            for jin = 1:length(TFleftList)
                v = XX(TFleftList(jin),:);
                P22 = 1/(D(TFleftList(jin)) + (v.*WW(jg,indw))*v' - (v.*WW(jg,indw))*XX(TFloc,:)'*G1*XX(TFloc,:)*(v.*WW(jg,indw))');
                P12 = -P22*G1*XX(TFloc,:)*(v.*WW(jg,indw))';
                P11 = G1 + P12*P12'/P22;
                G2 = [P11 P12; P12' P22];
                Atr0 = (YY(jg,:).*WW(jg,indw))*XX([TFlocList, TFleftList(jin)],:)'*G2;
                DeltaJ(jin) = Jnew - sum(WW(jg,indw).*(YY(jg,:) - Atr0*XX([TFlocList, TFleftList(jin)],:)).^2,2) + sum(Atr0.^2*diag(D([TFlocList, TFleftList(jin)])),2);
            end

            %Link scoring
            [Kscale, imax] = max(DeltaJ);
            CF{jg}(1,TFleftList) = CF{jg}(1,TFleftList) + (DeltaJ/Kscale);
            CF{jg}(2,TFleftList) = CF{jg}(2,TFleftList) + 1;
            
            %Add the one with highest importance to the regressor set TFloc
            iToAdd = TFleftList(imax);
            TFleft(iToAdd) = false;
            TFleftList = [TFleftList(1:imax-1); TFleftList(imax+1:end)];
            TFlocList = [TFlocList(1:sum(TFloc(1:iToAdd))) iToAdd  TFlocList(sum(TFloc(1:iToAdd))+1:end)];
            TFloc(iToAdd) = true;
  
            %Add also the mutated gene
            if ismember(iToAdd,perturbations.mutGenes)
                iMut = find(perturbations.mutGenes == iToAdd);
                TFloc(ndim+npert+iMut) = true;
                ia = sum(ndim+npert+iMut > TFloclist);
                TFlocList = [TFlocList(1:ia) ndim+npert+iMut TFlocList(ia+1:end)];
            end
                
        else %Backward greedy step

            %Check regressor importance in the current model
            DeltaJ = zeros(1,length(Atemp)-sum(TFloc(ndim+npert+1:end))); 
            for jin = 1:length(Atemp)-sum(TFloc(ndim+npert+1:end)) 
                
                if ismember(TFlocList(jin),perturbations.mutGenes)
                    iMut = find(perturbations.mutGenes == TFlocList(jin));
                    mutLoc = find(TFlocList == ndim+npert+iMut); 
                    G2 = G1([1:jin-1 jin+1:mutLoc-1 mutLoc+1:size(G1,1)],[1:jin-1 jin+1:mutLoc-1 mutLoc+1:size(G1,1)]) - G1([1:jin-1 jin+1:mutLoc-1 mutLoc+1:size(G1,1)],[jin mutLoc])*G1([jin, mutLoc],[jin,mutLoc])^-1*G1([1:jin-1 jin+1:mutLoc-1 mutLoc+1:size(G1,1)],[jin mutLoc])';
                    Atr0 = (YY(jg,:).*WW(jg,indw))*XX(TFlocList([1:jin-1 jin+1:mutLoc-1 mutLoc+1:length(TFlocList)]),:)'*G2;  
                    Atr0 = [Atr0(1:jin-1) 0 Atr0(jin:mutLoc-2) 0 Atr0(mutLoc-1:end)];
                else
                    G2 = [G1(1:jin-1,1:jin-1), G1(1:jin-1,jin+1:end); G1(jin+1:end,1:jin-1), G1(jin+1:end,jin+1:end)] - [G1(1:jin-1,jin); G1(jin+1:end,jin)]*[G1(1:jin-1,jin); G1(jin+1:end,jin)]'/G1(jin,jin);               
                    Atr0 = (YY(jg,:).*WW(jg,indw))*XX(TFlocList([1:jin-1 jin+1:length(TFlocList)]),:)'*G2;
                    Atr0 = [Atr0(1:jin-1) 0 Atr0(jin:end)];
                end
                DeltaJ(jin) = sum(WW(jg,indw).*(YY(jg,:) - Atr0*XX(TFloc,:)).^2,2) + sum(Atr0.^2*diag(D(TFloc)),2) - Jnew;

                if TFlocList(jin) == jg
                    indTarget = jin;
                end
            end
               
            
            %Link scoring
            if length(DeltaJ) > 1
                Kscale = .95*max(DeltaJ([1:indTarget-1 indTarget+1:length(DeltaJ)])) + .05*DeltaJ(indTarget);
            else
                Kscale = DeltaJ;
            end
           
            CB{jg}(1,TFlocList(1:end-sum(TFloc(ndim+npert+1:end)))) = CB{jg}(1,TFlocList(1:end-sum(TFloc(ndim+npert+1:end)))) + min(DeltaJ/Kscale,1);
            CB{jg}(2,TFlocList(1:end-sum(TFloc(ndim+npert+1:end)))) = CB{jg}(2,TFlocList(1:end-sum(TFloc(ndim+npert+1:end)))) + 1;

            %Mutation scoring
            for jmut = 1:length(perturbations.mutGenes)
                if TFloc(ndim+npert+jmut) 
                    jaux = find(TFlocList == ndim+npert+jmut); 
                    G2 = [G1(1:jaux-1,1:jaux-1), G1(1:jaux-1,jaux+1:end); G1(jaux+1:end,1:jaux-1), G1(jaux+1:end,jaux+1:end)] - [G1(1:jaux-1,jaux); G1(jaux+1:end,jaux)]*[G1(1:jaux-1,jaux); G1(jaux+1:end,jaux)]'/G1(jaux,jaux);               
                    Atr0 = (YY(jg,:).*WW(jg,indw))*XX(TFlocList([1:jaux-1 jaux+1:length(TFlocList)]),:)'*G2;
                    Atr0 = [Atr0(1:jaux-1) 0 Atr0(jaux:end)];
                    DeltaJmut = sum(WW(jg,indw).*(YY(jg,:) - Atr0*XX(TFloc,:)).^2,2) + sum(Atr0.^2*diag(D(TFloc)),2) - Jnew;
                    CB{jg}(1,ndim+npert+jmut) = CB{jg}(1,ndim+npert+jmut) + DeltaJmut/Kscale; 
                    CB{jg}(2,ndim+npert+jmut) = CB{jg}(2,ndim+npert+jmut) + 1;
                end
            end
            
           
            %Prevent the diagonal element being excluded
            DeltaJ(indTarget) = DeltaJ(indTarget)+1e10;

            %Remove the one with lowest importance from the regressor set TFloc
            [~,imin] = min(DeltaJ);
            iToRemove = TFlocList(imin);
            TFlocList = [TFlocList(1:imin-1) TFlocList(imin+1:end)];
            TFloc(iToRemove) = false;
            if ismember(iToRemove,perturbations.mutGenes)
                iMut = find(perturbations.mutGenes == iToRemove);
                TFloc(ndim+npert+iMut) = false; 
                imin = find(TFlocList == ndim+npert+iMut); 
                TFlocList = [TFlocList(1:imin-1) TFlocList(imin+1:end)];
            end

        end
    end
end
      
%Calculate link scores
GRN = zeros(ndim,ndim+npert+nmut);
auxNet = zeros(ndim,ndim);
for jout = 1:ndim
    for jin = 1:ndim+npert+nmut
        if jin <= ndim && CF{jout}(2,jin) > 0
            auxNet(jout,jin) = CF{jout}(1,jin)/CF{jout}(2,jin);
        end
        if CB{jout}(2,jin) > 0
            GRN(jout,jin) = CB{jout}(1,jin)/CB{jout}(2,jin);
        end
    end
end


%Fill in the zeros in GRN using the scores from the forward sweep 
for jg = 1:size(GRN,1)
    zz = GRN(jg,1:ndim) == 0;
    GRN(jg,zz) = 0.9*auxNet(jg,zz)/(max(auxNet(jg,zz))+1e-10)*min(GRN(jg,~zz));
end

%Check if signed links are asked for
if opts.signed
    GRN(:,1:ndim) = GRN(:,1:ndim).*sign(A(:,1:ndim));
end

if opts.par > 1
    delete(gcp('nocreate'))
end

