%%第二种方法


%初始化
k = 0;
nA = 34;
nB = 43;
nC = 19;
n = nA+nB+nC;
Aeq = [zeros(1,n),ones(1,n)];
beq = 5000;
l = [0.04*ones(n,1),10*ones(n,1)];
u = [0.15*ones(n,1),100*ones(n,1)];
% x0 = zeros(n,1);
x0 = [0.04*ones(n,1);10*ones(n,1)];
gf0 = gradf(x0);
Hes0 = Hession(x0);
%Choose the optimization algorithm:'dual-simplex' (default)'interior-point-legacy''interior-point'
fun = @(x)gf0'*x+x'*Hes0*x;
% 'active-set', 'interior-point', 'sqp', 'trust-region-reflective', or 'sqp-legacy'。
options = optimoptions('fmincon','Algorithm','interior-point','MaxIter',8000);
x1 = fmincon(fun,x0,[],[],Aeq,beq,l,u,[],options);

maxiter = 3000;
tol = 1e-5;

%循环
while k<maxiter & norm(x1-x0)>tol
x0 = x1;
gf0 = gradf(x0);
Hes0 = Hession(x0);
x1 = fmincon(fun,x0,[],[],Aeq,beq,l,u,[],options);
k = k+1;
end
disp('第二种解法的解：')
disp(x1)


function Hes = Hession(x)
alpha = 1;
% a b c d 为delta的系数
a = [640.9 552.8 504.7];
b = [-258.6 -225.1 -207.4];
c = [37.97 33.99 32.16];
d = [-1.121 -1.017 -0.9735];
load('evalue.mat')
% e = zeros(size(x));
deltaA = @(x)a(1)*x.^3+b(1)*x.^2+c(1)*x+d(1);
deltaB = @(x)a(2)*x.^3+b(2)*x.^2+c(2)*x+d(2);
deltaC = @(x)a(3)*x.^3+b(3)*x.^2+c(3)*x+d(3);
graddltA = @(x)3*a(1)*x.^2+2*b(1)*x+c(1);
graddltB = @(x)3*a(2)*x.^2+2*b(2)*x+c(2);
graddltC = @(x)3*a(3)*x.^2+2*b(3)*x+c(3);
grad2dltA = @(x)6*a(1)*x+2*b(1);
grad2dltB = @(x)6*a(2)*x+2*b(2);
grad2dltC = @(x)6*a(3)*x+2*b(3);
nA = 34;
nB = 43;
nC = 19;
n = nA+nB+nC;
gf = zeros(2*n,1);
IA = [1:nA];
IB = [nA+1:nA+nB];
IC = [nA+nB+1:nA+nB+nC];

Hes1 = diag(-2*graddltA(x(IA)).*x(n+IA)-grad2dltA(x(IA)).*(x(IA)-alpha*e(IA)).*x(n+IA));
Hes2 = diag(-2*graddltB(x(IB)).*x(n+IB)-grad2dltB(x(IB)).*(x(IB)-alpha*e(IB)).*x(n+IB));
Hes3 = diag(-2*graddltC(x(IC)).*x(n+IC)-grad2dltC(x(IC)).*(x(IC)-alpha*e(IC)).*x(n+IC));
Hes4 = diag(1-deltaA(x(IA))-graddltA(x(IA)).*(x(IA)-alpha*e(IA)));
Hes5 = diag(1-deltaB(x(IB))-graddltB(x(IB)).*(x(IB)-alpha*e(IB)));
Hes6 = diag(1-deltaC(x(IC))-graddltC(x(IC)).*(x(IC)-alpha*e(IC)));
Hes7 = diag(1-deltaA(x(IA))-graddltA(x(IA)).*(x(IA)-alpha*e(IA)));
Hes8 = diag(1-deltaB(x(IB))-graddltB(x(IB)).*(x(IB)-alpha*e(IB)));
Hes9 = diag(1-deltaC(x(IC))-graddltC(x(IC)).*(x(IC)-alpha*e(IC)));
Hes = zeros(2*n,2*n);
Hes(1:n,1:n)=blkdiag(Hes1,Hes2,Hes3);
Hes(1:n,n+1:2*n)=blkdiag(Hes7,Hes8,Hes9);
Hes(n+1:2*n,1:n) = blkdiag(Hes4,Hes5,Hes6);
end



function gf = gradf(x)
alpha = 1;
% a b c d 为delta的系数
a = [640.9 552.8 504.7];
b = [-258.6 -225.1 -207.4];
c = [37.97 33.99 32.16];
d = [-1.121 -1.017 -0.9735];
load('evalue.mat')
% e = zeros(size(x));
deltaA = @(x)a(1)*x.^3+b(1)*x.^2+c(1)*x+d(1);
deltaB = @(x)a(2)*x.^3+b(2)*x.^2+c(2)*x+d(2);
deltaC = @(x)a(3)*x.^3+b(3)*x.^2+c(3)*x+d(3);
graddltA = @(x)3*a(1)*x.^2+2*b(1)*x+c(1);
graddltB = @(x)3*a(2)*x.^2+2*b(2)*x+c(2);
graddltC = @(x)3*a(3)*x.^2+2*b(3)*x+c(3);
nA = 34;
nB = 43;
nC = 19;
n = nA+nB+nC;
gf = zeros(2*n,1);
IA = [1:nA];
IB = [nA+1:nA+nB];
IC = [nA+nB+1:nA+nB+nC];

% gf(IA) = -((1-deltaA(x(IA))).*x(n+IA)-graddltA(x(IA)).*(x(IA)-alpha*e(IA)).*x(n+IA));
% gf(IB) = -((1-deltaA(x(IB))).*x(n+IB)-graddltA(x(IB)).*(x(IB)-alpha*e(IB)).*x(n+IB));
% gf(IC) = -((1-deltaA(x(IC))).*x(n+IC)-graddltA(x(IC)).*(x(IC)-alpha*e(IC)).*x(n+IC));
% gf(n+IA) = -((1-deltaA(x(IA))).* (x(IA)-alpha*e(IA)));
% gf(n+IB) = -((1-deltaA(x(IB))).* (x(IB)-alpha*e(IB)));
% gf(n+IC) = -((1-deltaA(x(IC))).* (x(IC)-alpha*e(IC)));


gf(IA) = -graddltA(x(IA)).*e(IA).*x(n+IA);
gf(IB) = -graddltA(x(IB)).*e(IB).*x(n+IB);
gf(IC) = -graddltA(x(IC)).*e(IC).*x(n+IC);
gf(n+IA) = ((1-deltaA(x(IA))).* e(IA));
gf(n+IB) = ((1-deltaA(x(IB))).* e(IB));
gf(n+IC) = ((1-deltaA(x(IC))).* e(IC));
end