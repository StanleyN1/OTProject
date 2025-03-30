function [transport_cost, reg_cost, M, iteration_count, u1] = OTsolver(mu0,mu1,C, epsilon, uInit)

mu0 = mu0(:);
mu1 = mu1(:);

if abs(size(mu0,1)-size(C,1))+abs(size(mu1,1)-size(C,2)) >= 1
    disp('dimension error')
    mu0dim = size(mu0)
    mu1dim = size(mu1)
    Cdim = size(C)
end


K = exp(-C/epsilon);

%Initialise the dual variables 
if exist('uInit')
    u1 = uInit(:);
else
    u1 = mu1;
end
u0 = mu0;
u0_old = 100;
k = 0;

while norm(log(u0_old)-log(u0)) > 1e-3 && k < 10000
    u0_old = u0;
    u0 = mu0./(K*u1);
    u1 = mu1./(K'*u0);
    k = k+1;
end

iteration_count = k;
M = diag(u0)*K*diag(u1);
transport_cost = sum(sum(M.*C)); 

%NaNs are produced for zero entries that should be ignored
EE = M(:).*log(M(:));
reg_cost = sum(EE(~isnan(EE)));




