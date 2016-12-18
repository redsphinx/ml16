function [w, i, trainerror] = perceptronlearn(X, c, eta, maxiter)
% PERCEPTRONLEARN Completes the perceptron learning procedure.
% X is an NxD matrix containing the data set, and they have target classes
% c as an Nx1 vector with values -1 or 1. Uses learning rate eta and
% continues until convergence or maxiter iterations, whichever comes first.
% Returns the final weights w, the number of iterations taken i,
% and the training error as a fraction of 1.

n = size(X, 2)-1;

% Precision parameters
tolerance = 0.5/eta; % bound for the total weight change per iteration to fall under for "convergence"
convergencecount = 5; % number of times in a row negligible weight changes are needed for "convergence"
batchSize = 256; % max. number of training samples used per iteration

% Initialise
w = randn(1,n+1); % weights w(1:n) threshold w(n+1)
batchSize = min(size(X, 1), batchSize);
nconv = 0;

% Simulate using mini-batch gradient descent
for i=1:maxiter
    prevw = w;
    
    % Get mini-batch from data set
	[batch, batchis] = datasample(X, batchSize, 'Replace', false);
	
	% Apply the perceptron learning rule
	% N.B. This learning rule differs slightly from the one used in
	% the lecture
    w = w + eta * (c(batchis) - perceptronforward(batch, w))' * batch;
    
    if sum((prevw-w).^2) < tolerance
        nconv = nconv + 1;
        if nconv >= convergencecount
            break;
        end
    else
        nconv = 0;
    end
end

if nargout > 2
    trainerror = mean(perceptronforward(X, w) ~= c);
end


end