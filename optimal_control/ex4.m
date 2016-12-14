close all
fig = figure;

nus = 1:1:20;
dt = 0.01;
Ts = [1 2 5 10];

for iT=1:numel(Ts)
    T = Ts(iT);
    subplot(2,2,iT)

    for n=1:numel(nus)
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

            dxi = randn(1) * sqrt(nu*dt);
            x = x + uhist(i)*dt + dxi;
    end

    % plot(ts, xhist, ts, uhist)
    nuspread = 2*(max(nus) - min(nus));
    color = @(i) [0.5+(i-min(nus))/nuspread, 0.5-(i-min(nus))/nuspread, 0.5-(i-min(nus))/nuspread];


    plot(ts, xhist, 'Color', color(nu))
    hold on

    % hold off
    % legend('x', 'zero axis', 'targets', 'Location', 'southwest')
    % title(strcat({'Optimal Control, nu='},num2str(nu), {', T='},num2str(T)))
    end

    plot([0 T], [0 0], '--', 'Color', [0.5 0.5 0.5])
    plot([0 T], [1 1], '--', 'Color', [1 0 0])
    plot([0 T], [-1 -1], '--', 'Color', [1 0 0])
    xlabel('t')

    xlim([T-1 T]);
end

suptitle('suuuper')

cmap = reshape(color(nus), [numel(nus) 3]);
colormap(cmap);
cbh = colorbar('peer', gca, [0.924414348462665 0.106870229007634 0.0183016105417277 0.743002544529263]);
set(cbh, 'YTickLabel', char(arrayfun(@num2str, get(cbh, 'YTick') + min(nus)-1, 'UniformOutput', false)));
ylabel(cbh, 'nu')