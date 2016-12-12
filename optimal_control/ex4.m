nu = 10;
T = 5;
dt = 0.01;

u = @(x,t) (tanh(x ./ (nu*(T-t))) - x) ./ (T-t);

rng(0)
x = 0;
ts = 0:dt:T;

xhist = zeros(1, numel(ts));
uhist = NaN(1, numel(ts));

for i=1:numel(ts)    
    t = ts(i);
    xhist(i) = x;
    uhist(i) = u(x,t);
    
    dxi = randn(1) * sqrt(nu*dt);
    x = x + uhist(i)*dt + dxi;
end

plot(ts, xhist, ts, uhist)
hold on
plot([0 T], [0 0], '--', 'Color', [0.5 0.5 0.5]);
hold off
legend('x', 'u')
xlabel('t')
