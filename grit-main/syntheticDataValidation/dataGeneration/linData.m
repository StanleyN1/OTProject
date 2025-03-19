function [scdata,Tgrid,GT] = linData(simOpts,Tgrid)


%Get the system corresponding to the 
[A,b,GT,x0,x1] = getLinSystem(simOpts.seed,false);
dt = simOpts.dt;

clear('tim')
if exist('Tgrid')
    tim(1,:) = Tgrid;
    tim(2,:) = (0:length(Tgrid)-1)*simOpts.nc + 1;
    tim(3,:) = (1:length(Tgrid))*simOpts.nc;
else
    tim(1,:) = simOpts.d*(0:simOpts.T-1);
    tim(2,:) = (0:simOpts.T-1)*simOpts.nc + 1;
    tim(3,:) = (1:simOpts.T)*simOpts.nc;
end

%Initial distribution covariance
rng(simOpts.seed)
P = randn(10,10);
P = P*P';
dp = diag(P).^-.5;
P = (x0.^.5.*((dp.*P).*dp')).*x0'.^.5+.3*eye(10);
CP = chol(P)';

%Steady-state is used for noise intensity scaling
xss = [3.9604 1.3036 4.4812 2.3148 1.7788 4.1574 1.4179 2.2551 4.3720 4.6660]';
Dnoise = (-2*diag(A).*xss).^.5;
if isfield(simOpts,'noisetype') && strcmp(simOpts.noisetype,'uniform')
    Dnoise = mean(Dnoise)*ones(10,1);
end

%Initial distribution
Y = x0 + simOpts.Ks*CP*randn(10,tim(3,end));

for jtim = 2:size(tim,2)
    
    if simOpts.cont
        %Discretisation
        nt = round(tim(1,jtim)/dt);
        for jt = 1:nt
            for jcell = tim(2,jtim):tim(3,jtim)
                dx = A*Y(:,jcell) + b;
                Y(:,jcell) = Y(:,jcell) + dt*dx + simOpts.Kn*dt^.5/2^.5*Dnoise.*sum(randn(10,2),2);
                Y(:,jcell) = max(Y(:,jcell),0);
            end
        end
    else
        for jt = 2:jtim
            %Discrete-time system
            Y(:,tim(2,jtim):tim(3,jtim)) = (eye(size(A,1)) + (tim(1,jt)-tim(1,jt-1))*A) * Y(:,tim(2,jtim):tim(3,jtim)) + (tim(1,jt)-tim(1,jt-1))*b + simOpts.Kn*(tim(1,jt)-tim(1,jt-1))^.5*Dnoise.*randn(size(A,1),tim(3,jtim)-tim(2,jtim)+1);
        end
    end
    
end


%Put data into required format
Tgrid = tim(1,:);
clear('scdata')
for jt = 1:size(tim,2)
    scdata{jt} = Y(:,tim(2,jt):tim(3,jt));
end
