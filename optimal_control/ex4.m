nus = 10:1:20;
T = 5;
dt = 0.01;
legendstrings = {};

for n=1:numel(nus)
% nu = 0.1;
nu =  nus(n);
u = @(x,t) (tanh(x ./ (nu*(T-t))) - x) ./ (T-t);
rng(1)
x = 0;
ts = 0:dt:T;

xhist = zeros(1, numel(ts));
uhist = NaN(1, numel(ts));

for i=1:numel(ts)    
    t = ts(i);
    xhist(i) = x;
    uhist(i) = u(x,t);
    
    dxi = randn(1) * sqrt(nu*dt)
    x = x + uhist(i)*dt + dxi;
end

% plot(ts, xhist, ts, uhist)
color = @(i) [0.5+i/20, 0.5-i/20, 0.5-i/20];


plot(ts, xhist, 'Color', color(nu - 10))
hold on

% hold off
% legend('x', 'zero axis', 'targets', 'Location', 'southwest')
% title(strcat({'Optimal Control, nu='},num2str(nu), {', T='},num2str(T)))
end

plot([0 T], [0 0], '--', 'Color', [0.5 0.5 0.5])
plot([0 T], [1 1], '--', 'Color', [1 0 0])
plot([0 T], [-1 -1], '--', 'Color', [1 0 0])
xlabel('t')
