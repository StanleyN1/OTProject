function [scdata,Tgrid,GT] = MutData(simOpts,Tgrid,MutC,pertC)

if ~exist('MutC')
    MutC = ones(3,1);
end
if ~exist('pertC')
    MutC = ones(10,1);
end

dt = simOpts.dt;

clear('tim')
if exist('Tgrid') && length(Tgrid) > 0
    tim(1,:) = Tgrid;
    tim(2,:) = (0:length(Tgrid)-1)*simOpts.nc + 1;
    tim(3,:) = (1:length(Tgrid))*simOpts.nc;
else
    tim(1,:) = simOpts.d*(0:simOpts.T-1);
    tim(2,:) = (0:simOpts.T-1)*simOpts.nc + 1;
    tim(3,:) = (1:simOpts.T)*simOpts.nc;
end


[~,~,GT,x0,x1] = getLinSystem(mod(simOpts.seed-1,10)+1,false);


%Initial distribution covariance
rng(mod(simOpts.seed-1,10)+1)
P = randn(10,10);
P = P*P';
dp = diag(P).^-.5;
P = (x0.^.5.*((dp.*P).*dp')).*x0'.^.5+.3*eye(10);
CP = chol(P)';


rng(simOpts.seed)

%Initial distribution
Y = max(x0 + simOpts.Ks*CP*randn(10,tim(3,end)),0);

for jtim = 2:size(tim,2)
    nt = round(tim(1,jtim)/dt);
    for jt = 1:nt
        for jcell = tim(2,jtim):tim(3,jtim)
            dx = fnetMut(Y(:,jcell),MutC,pertC);
            Y(:,jcell) = Y(:,jcell) + dt*dx*[-1; 1] + simOpts.Kn*dt^.5*sum(dx.^.5.*randn(10,2),2);
            Y(:,jcell) = max(Y(:,jcell),0);
        end
    end
end


%Put data into required format
Tgrid = tim(1,:);
clear('scdata')
for jt = 1:size(tim,2)
    scdata{jt} = Y(:,tim(2,jt):tim(3,jt));
end



