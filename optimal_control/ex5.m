clear

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
xs = -2:0.1:2;
vs = -2:0.1:2;

% Relevant quantities
L = @(x) -1 -0.5*(tanh(2*x + 2) - tanh(2*x - 2));
Lderiv = @(x) sech(2 - 2*x).^2 - sech(2*x + 2).^2;
Fg = @(x) (-g*Lderiv(x)) ./ sqrt(1 + Lderiv(x).^2);

% Plot valley, slope, gravity
figure
xs = -valleywidth:0.01:valleywidth;
plot(xs, L(xs), xs, Lderiv(xs), xs, Fg(xs));
legend('valley', 'slope', 'gravity')
xlabel('x')

% Prepare measurements
constant_numel_xs = numel(xs);
constant_numel_vs = numel(vs);

phihist = zeros(constant_numel_xs, constant_numel_vs, numtries);

parfor ix = 1:constant_numel_xs
    for iv = 1:constant_numel_vs
        fprintf('.')

        maxhist = NaN(1, numtries);

        for seed=1:numtries
            rng(seed+238765); % reproducability

            % Initialisation
            x = xs(ix);
            v = vs(iv);
            ts = 0:dt:T;

            xhist = NaN(1, numel(ts));
            vhist = NaN(1, numel(ts));

            for i=1:numel(ts)
                % Execute dynamics
                
                t = ts(i);

                u = 0;
                dxi = randn(1) * sqrt(nu*dt);

                xhist(i) = x;
                vhist(i) = v;

                x = x + v*dt;
                v = Fg(x)*dt + u*dt + dxi;
            end

            phihist(ix, iv, seed) = A * (abs(x) > valleywidth); 
        end
    end
end

% Compute cost-to-go J
lambda = 0.1 * nu;
Jtable = -lambda * log(sum(exp(-phihist/lambda), 3)/numtries);

% Plot cost-to-go J
[X,V] = ndgrid(xs,vs);
surf(X, V, Jtable)
xlabel('x')
ylabel('v')
zlabel('J')
title('Cost-to-go J for different initial positions, velocities')