function [A,b,GT,x0,x1] = getLinSystem(seed,showPlot)

if ~exist('showPlot')
    showPlot = false;
end
rng(seed)


%System A,b obtained by linearising the function fnet around its steady
%state given by xss.


% Steady state obtained by this procedure
% fun = @(x)(sum((fnet(x)*[1; -1]).^2));
% xss = ones(10,1);
% for jt = 1:3
%     xss = fminsearch(fun,xss);
% end


xss = [3.9604 1.3036 4.4812 2.3148 1.7788 4.1574 1.4179 2.2551 4.3720 4.6660]';

%Parameters of the fnet function
a1 = 1.5; a2 = 1.5; a3 = .4; a4 = 1.5; a5 = 1.5; a6 = 1.5; a7 = 2; a8 = .8; a9 = .5; a10 = 1.2;
k1 = 2.0991; k2 = 3.6469; k3 = 2.5882; k4 = 3.9127; k5 = 1.5617; k6 = 3.6369; k7 = 2.2910; k8 = 3.7228; k9 = 1.0370; k10 = 3.2482;
v1 = 0.4524; v2 = 0.5124; v3 = 0.4377; v4 = 0.0479; v5 = 0.3310; v6 = 0.1265; v7 = 0.3932; v8 = 0.4238; v9 = 0.1570; v10 = 0.3040;
c1 = 0.2015; c2 = 0.2318; c3 = 0.8641; c4 = 0.9068; c5 = 0.2069; c6 = 0.4267; c7 = 0.0411; c8 = 1.5239; c9 = 0.0628; c10 = 0.3638;
d2 = 0.6667; d7 = 0.4545;

A = -diag([a1 a2 a3 a4 a5 a6 a7 a8 a9 a10]);

%Gene 1
A(1,7) = (k1-c1*v1)/(1+c1*(xss(7)+xss(10)))^2;
A(1,10) = (k1-c1*v1)/(1+c1*(xss(7)+xss(10)))^2;

%Gene 2
A(2,1) = (k2-c2*v2)/(1+c2*xss(1))^2/(1+d2*xss(3));
A(2,3) = -d2*(k2*xss(1)+v2)/(1+c2*xss(1))/(1+d2*xss(3))^2;

%Gene 3
A(3,2) = (k3-c3*v3)/(1+c3*xss(2))^2;

%Gene 4
A(4,3) = (k4-c4*v4)/(1+c4*xss(3))^2;

%Gene 5
A(5,4) = (k5-c5*v5)/(1+c5*xss(4))^2;

%Gene 6
A(6,3) = (k6-c6*v6)/(1+c6*(xss(3)+xss(5)))^2;
A(6,5) = (k6-c6*v6)/(1+c6*(xss(3)+xss(5)))^2;

%Gene 7
A(7,6) = (k7-c7*v7)/(1+c7*xss(6))^2/(1+d7*xss(9));
A(7,9) = -d7*(k7*xss(6)+v7)/(1+c7*xss(6))/(1+d7*xss(9))^2;

%Gene 8
A(8,7) = (k8-c8*v8)/(1+c8*xss(7))^2;

%Gene 9
A(9,8) = (k9-c9*v9)/(1+c9*xss(8))^2;

%Gene 10
A(10,9) = (k10-c10*v10)/(1+c10*xss(9))^2;


GT = abs(A) > .01;
GT = GT - diag(diag(GT));


b = -A*xss;
x0 = 2*rand(10,1).*xss;
x1 = (.85 + .3*(rand(10,1) > .5)).*x0;


% Example trajectory
if showPlot
    x = zeros(10,801);
    x(:,1) = x0;
    dt = .01;

    Ad = expm(A*dt);
    for j = 2:801
        x(:,j) = Ad*x(:,j-1) + dt*b + 0*0.1*dt^.5*randn(10,1);
    end

    figure; hold on; grid on;
    for j=1:10
        plot(dt*(0:800),x(j,:),'LineWidth',1)
    end
end








