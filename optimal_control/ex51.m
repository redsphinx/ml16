clear
clc
clf

% Arena parameters
valleywidth = 2;
Tlist = [100000];
dt = 0.1;

% Problem parameters
nus = 20;
g = 1;
R = 1;
A = -1;

% Plotting parameters
numtries = 100;
xs = -2:0.1:2;
vs = -2:0.1:2;

% Relevant quantities
L = @(x) -1 - 0.5 * (tanh(2*x + 2) - tanh(2*x - 2));
Lderiv = @(x) sech(2 - 2*x).^2 - sech(2*x + 2).^2;
Fg = @(x) (-g*Lderiv(x)) ./ sqrt(1 + Lderiv(x).^2);

%{
xs = -valleywidth:0.01:valleywidth;
plot(xs, L(xs), xs, Lderiv(xs), xs, Fg(xs));
legend('L', 'L''', 'Fg')
xlabel('x')
%}

% Prepare measurements


constant_numel_xs = numel(xs);
constant_numel_vs = numel(vs);

phihist = zeros(constant_numel_xs, constant_numel_vs, numtries);

maxhist = NaN(1, numtries);

figure(1)

for t_idx = 1:numel(Tlist)
    for seed=1:numtries
        nu = 0.1;
        T = Tlist(t_idx);
        rng(seed+238765);
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
        if max(abs(xhist)) >= 2
            color = 'b';
        else
            color='r';
        end
    %     maxhist(seed) = max(abs(xhist));
    %     subplot(1,2,1)
    %     
        subplot(2,2,t_idx);
        hold on;
        plot(ts, xhist, 'Color', color);
        xlim([0 T]);
        xlabel(strcat('T=', num2str(T)));
    %     subplot(1,2,2)
    %     hold on
    %     plot(ts, vhist, 'Color', color);
    %     phihist(ix, iv, seed) = A * (abs(x) > valleywidth); 


    end
end
hold off
% figure(2)
% boxplot(maxhist)


% lambda = 0.1 * nu;
% Jtable = -lambda * log(sum(exp(-phihist/lambda), 3)/numtries);
% 
% [X,V] = ndgrid(xs,vs);
% surf(X, V, Jtable)
% xlabel('x')
% ylabel('v')
% zlabel('J')
