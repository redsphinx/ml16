% Parameter settings
maxgamma = 5;
numgammas = 1000;
epsilon = 0.0002;
T = 1000;
tolerance = 1e-6;

% example 1 from Zhao and Yu 2006 (taken from correlated_data.m)
n=3;
p=1000;      % train data size
w1=[2,3,0];  % example 1a
w2=[-2,3,0]; % example 1b
sigma=1;
x(1:2,:)=randn(2,p);
x(3,:)=2/3*x(1,:)+2/3*x(2,:)+1/3*randn(1,p);
y1=w1*x+randn(1,p);
y2=w2*x+randn(1,p);
b1 = x * y1' / p;
b2 = x * y2' / p;

% Standardise data set
x = zscore(x, 1, 2);
y1 = zscore(y1, 1, 2);
y2 = zscore(y2, 1, 2);

% Start main script
gammas = logspace(log10(maxgamma), log10(maxgamma*epsilon), numgammas+1);
betas1 = lasso_coord_descent(x, y1, gammas, T, tolerance);
betas2 = lasso_coord_descent(x, y2, gammas, T, tolerance);
norms1 = sum(abs(betas1), 2);
norms2 = sum(abs(betas2), 2);

% Plots
legendstrings = cell(n, 1);
for in=1:n
    if in < n
        plot(norms1, betas1(:,in));
    else
        plot(norms1, betas1(:,in), '--');
    end
    legendstrings{in} = sprintf('\\beta_%i (b_%i = %g)', in, in, b1(in));
    hold on
end
hold off
xlabel('||\beta||_1')
ylabel('coefficients')
title('LASSO-derived coefficients for different values of \gamma')
legend(legendstrings, 'Location', 'NorthWest')

figure
for in=1:n
    if in < n
        plot(norms2, betas2(:,in));
    else
        plot(norms2, betas2(:,in), '--');
    end
    legendstrings{in} = sprintf('\\beta_%i (b_%i = %g)', in, in, b2(in));
    hold on
end
hold off
xlabel('||\beta||_1')
ylabel('coefficients')
title('LASSO-derived coefficients for different values of \gamma')
legend(legendstrings, 'Location', 'NorthWest')