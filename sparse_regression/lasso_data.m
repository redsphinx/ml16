% Parameter settings
maxgamma = 10;    % numgammas gamma values in the range
numgammas = 1000;  % maxgamma to epsilon*maxgamma
epsilon = 0.0001; % are tested for both ridge regr. and lasso.
T = 2500;         % max. #iterations lasso
tolerance = 1e-6; % L2-norm threshold for convergence

% Load data set
xtrain = load('data1_input_train');
xtest = load('data1_input_val');
ytrain = load('data1_output_train');
ytest = load('data1_output_val');
[n,p] = size(xtrain);

% Start cross-validation
cvp = cvpartition(p, 'k', 10);
sselasso = zeros(numgammas+1, 1);
sseridge = zeros(numgammas+1, 1);

for k=1:cvp.NumTestSets
    % Prepare data set for this fold
    fprintf('Iteration %i...\n', k);
    [xtraink, xmu, xsigma] = zscore(xtrain(:, cvp.training(k)), 1, 2);
    [ytraink, ymu, ysigma] = zscore(ytrain(:, cvp.training(k)), 1, 2);
    ptestk = cvp.TestSize(k);
    xtestk = (xtest(:, cvp.test(k)) - repmat(xmu, [1 ptestk])) ./ repmat(xsigma, [1 ptestk]);
    ytestk = (ytest(:, cvp.test(k)) - repmat(ymu, [1 ptestk])) ./ repmat(ysigma, [1 ptestk]);
    gammas = logspace(log10(maxgamma), log10(maxgamma*epsilon), numgammas+1);

    % Start main LASSO script
    betaslasso = lasso_coord_descent(xtraink, ytraink, gammas, T, tolerance);
    predslasso = betaslasso * xtestk;
    
    % Perform ridge regression
    betasridge = ridge(ytraink', xtraink', gammas)';
    predsridge = betasridge * xtestk;
    
    % Evaluate on validation set
    sselasso = sselasso + sum((predslasso - repmat(ytestk, [numgammas+1 1])) .^ 2, 2);
    sseridge = sseridge + sum((predsridge - repmat(ytestk, [numgammas+1 1])) .^ 2, 2);
end

% Derive CV-optimal parameters
[~,gammalasso] = min(sselasso);
[~,gammaridge] = min(sseridge);

% Prepare full data set
fprintf('Final evaluation...\n');
[xtrain, xmu, xsigma] = zscore(xtrain, 1, 2);
[ytrain, ymu, ysigma] = zscore(ytrain, 1, 2);
xtest = (xtest - repmat(xmu, [1 p])) ./ repmat(xsigma, [1 p]);
ytest = (ytest - repmat(ymu, [1 p])) ./ repmat(ysigma, [1 p]);

% Start main LASSO script
betaslasso = lasso_coord_descent(xtrain, ytrain, gammas, T, tolerance);
normslasso = sum(abs(betaslasso), 2);
predslasso = betaslasso(gammalasso,:) * xtest;

% Perform ridge regression
betasridge = ridge(ytrain', xtrain', gammas)';
normsridge = sum(abs(betasridge), 2);
predsridge = betasridge(gammaridge,:) * xtest;

% Evaluate on test set
mselasso = mean((predslasso - ytest) .^ 2, 2);
mseridge = mean((predsridge - ytest) .^ 2, 2);
fprintf('\nMSE Lasso: %g\nMSE ridge: %g\n\n', mselasso, mseridge);

% Plots
for in=1:n
    %semilogx(gammas, betaslasso(:,in));
    plot(normslasso, betaslasso(:,in));
    hold on
end
hold off
%xlabel('\gamma')
%set(gca, 'XDir', 'reverse')
xlabel('||\beta||_1')
ylabel('\beta_i')
title('LASSO-derived coefficients for different values of \gamma')

figure
for in=1:n
    semilogx(gammas, betasridge(:,in));
    %plot(normsridge, betasridge(:,in));
    hold on
end
hold off
xlabel('\gamma')
set(gca, 'XDir', 'reverse');
%xlabel('||\beta||_1')
ylabel('\beta_i')
title('Ridge regression coefficients for different values of \gamma')

figure
semilogx(gammas, sselasso/sum(cvp.TestSize), gammas, sseridge/sum(cvp.TestSize));
xlabel('\gamma')
set(gca, 'XDir', 'reverse');
ylabel('MSE')
title('Mean squared errors for different values of \gamma')
legend('Lasso', 'Ridge regression')
