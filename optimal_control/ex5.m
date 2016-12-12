Jtable = NaN(9);
xs = -2:0.5:2;
vs = -2:0.5:2;
phihist = NaN(9,9,numtries);
for ix = 1:numel(xs)
x = xs(ix);
for iv = 1:numel(vs)
v = vs(iv);
fprintf('.')
numtries = 100;
maxhist = NaN(1, numtries);
phihist = NaN(1, numtries);
for seed=1:numtries
% Arena parameters
xmin = -2;
xmax = 2;
T = 100;

% Simulation parameters
dt = 0.1;
rng(seed);

% Problem parameters
nu = 10;
g = 1;
R = 1;
A = -1;

% Relevant quantities
L = @(x) -1 -0.5*(tanh(2*x + 2) - tanh(2*x - 2));
Lderiv = @(x) -0.5*(2*sech(2*x + 2).^2 - 2*sech(2 - 2*x).^2);
Fg = @(x) (-g*Lderiv(x)) ./ sqrt(1 + Lderiv(x).^2);

%{
xs = xmin:0.01:xmax;
plot(xs, L(xs), xs, Lderiv(xs), xs, Fg(xs));
legend('L', 'L''', 'Fg')
xlabel('x')
%}

x = 0.5;
v = 0;
ts = 0:dt:T;

xhist = NaN(1, numel(ts));
vhist = NaN(1, numel(ts));

for i=1:numel(ts)
    t = ts(i);
    
    u = 0;
    dxi = randn(1) * sqrt(nu*dt);
    
    xhist(i) = x;
    vhist(i) = v;
    
    x = x + v*dt;
    v = Fg(x)*dt + u*dt + dxi;
end

%plot(ts, xhist);
%legend('x')
phihist(ix, iv, seed) = A * (abs(x) > 2); 

% maxhist(seed) = max(abs(xhist));
end

% boxplot(maxhist)
lambda = 0.1 * nu;
Jtable(ix, iv) = -lambda * log(sum(exp(-phihist(ix, iv, seed)/lambda))/numtries);

end
end

[X,V] = meshgrid(xs,vs);
surf(X, V, Jtable)