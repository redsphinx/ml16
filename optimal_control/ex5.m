clear
clc

% Arena parameters
valleywidth = 2;
T = 10;
dt = 0.1;

% Problem parameters
nu = 20;
g = 1;
R = 1;
A = -1;

% Plotting parameters
numtries = 10;
xs = -2:0.8:2;
vs = -2:1:2;

% Relevant quantities
L = @(x) -1 -0.5*(tanh(2*x + 2) - tanh(2*x - 2));
Lderiv = @(x) sech(2 - 2*x).^2 - sech(2*x + 2).^2;
Fg = @(x) (-g*Lderiv(x)) ./ sqrt(1 + Lderiv(x).^2);

%{
xs = -valleywidth:0.01:valleywidth;
plot(xs, L(xs), xs, Lderiv(xs), xs, Fg(xs));
legend('L', 'L''', 'Fg')
xlabel('x')
%}

% Prepare measurements
phihist = NaN(numel(xs), numel(vs), numtries);

for ix = 1:numel(xs)
    for iv = 1:numel(vs)
        fprintf('.')

        maxhist = NaN(1, numtries);

        for seed=1:numtries
            rng(seed+238765);

            x = xs(ix);
            v = vs(iv);
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
            phihist(ix, iv, seed) = A * (abs(x) > valleywidth); 

            % maxhist(seed) = max(abs(xhist));
        end

        % boxplot(maxhist)
    end
end

lambda = 0.1 * nu;
Jtable = -lambda * log(sum(exp(-phihist/lambda), 3)/numtries);

[X,V] = ndgrid(xs,vs);
surf(X, V, Jtable)
xlabel('x')
ylabel('v')
zlabel('J')
