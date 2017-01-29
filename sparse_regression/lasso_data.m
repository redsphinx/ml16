% Parameter settings
maxgamma = 1;
numgammas = 250;
epsilon = 0.001;
T = 1000;
tolerance = 1e-6;

% Load data set
xtrain = load('data1_input_train');
xtest = load('data1_input_val');
ytrain = load('data1_output_train');
ytest = load('data1_output_val');

% Standardise data set
[n,p] = size(xtrain);
[xtrain, xmu, xsigma] = zscore(xtrain, 1, 2);
[ytrain, ymu, ysigma] = zscore(ytrain, 1, 2);
xtest = (xtest - repmat(xmu, [1 p])) ./ repmat(xsigma, [1 p]);
ytest = (ytest - repmat(ymu, [1 p])) ./ repmat(ysigma, [1 p]);

% Start main script
gammas = logspace(log10(maxgamma), log10(maxgamma*epsilon), numgammas+1);
betas = lasso_coord_descent(xtrain, ytrain, gammas, T, tolerance);
norms = sum(abs(betas), 2);

% Plot
for in=1:n
    %semilogx(gammas, betas(:,in));
    plot(norms, betas(:,in));
    hold on
end
hold off
xlabel('||\beta||_1')
ylabel('coefficients')
title('LASSO-derived coefficients for different values of \gamma')