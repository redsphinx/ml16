function [W, b, i, trainerror] = mlplearn(X, t, nhiddens, eta, maxiter)
% MLPLEARN Completes the entire learning procedure for one multi-layer perceptron.
% X is an NxDi matrix containing the N D-dimensional elements from the data
% set, with corresponding target activations t in an NxDo matrix/vector.
% nhiddens is a (scalar or) vector containing the number of hidden units
% per layer; the output units that learn to predict t are not included here.
% The learning rate used is eta, and learning continues either until
% convergence, or for maxiter iterations, whichever arrives first.
% Returns the final weights W, biases b, the number of iterations
% taken i and the training error (as a fraction of 1)

% Precision parameters
tolerance = 1e-4; % maximum weight+bias change (as Eucl.dist.)
convergencecount = 5;
batchSize = 256;

nunits = [nhiddens(:)', size(t, 2)]; % add output units
numlayers = length(nunits);
b = cell(1, numlayers);
W = cell(1, numlayers);

% Randomly initialise network weights
ninputs = size(X, 2);
for k=1:numlayers
    % Reference for dividing factor: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    W{k} = randn(ninputs, nunits(k)) / sqrt(ninputs);
    b{k} = zeros(1, nunits(k));
    ninputs = nunits(k);
end

batchSize = min(size(X, 1), batchSize);
nconv = 0;
Wdir = NaN;
bdir = NaN;

for i=1:maxiter
    %{
    if mod(i, 100) == 0
        fprintf('%i/%i iterations done...\n', i, maxiter);
    end
    %}
    
    prevW = W;
    prevb = b;
    
    % Get mini-batch from the data set
    [batch, batchi] = datasample(X, batchSize, 'Replace', false);
    
    % Compute output per layer and back-propagate the errors
    [~, Y] = mlpforward(batch, W, b);
    [dEdW, dEdb] = mlpbackward(batch, t(batchi), W, b, Y);
    
    % --- IMPORTANT ---
    % These are update rules for the different learning algorithms.
    % Uncomment the one you want to try, but always have at most one
    % update step uncommented.
    
    %{
    % Regular grad.desc.
    [W, Wdir] = adjustby(W, dEdW, -eta);
    [b, bdir] = adjustby(b, dEdb, -eta);
    %}
    
    %{
    % Regular grad.desc. with weight decay (lambda=0.001)
    [W, Wdir] = adjustby(W, dEdW, -eta, W, -0.001);
    [b, bdir] = adjustby(b, dEdb, -eta, b, -0.001);
    %}
    
    % {
    % Regular grad.desc. with momentum (rho=0.9) <-- this one seems to perform best
    [W, Wdir] = adjustby(W, dEdW, -eta, Wdir, 0.9);
    [b, bdir] = adjustby(b, dEdb, -eta, bdir, 0.9);
    % }
    
    %{
    % Regular grad.desc. with momentum and weight decay (rho=0.9, lambda=0.001)
    [W, Wdir] = adjustby(W, dEdW, -eta, Wdir, 0.9, W, -0.001);
    [b, bdir] = adjustby(b, dEdb, -eta, bdir, 0.9, b, -0.001);
    %}
    
    %{
    % Conj.grad.desc.
    beta = cgdbeta(dEdW, prevdEdW, dEdb, prevdEdb);
    [W, Wdir] = adjustby(W, dEdW, -eta, Wdir, eta*beta);
    [b, bdir] = adjustby(b, dEdb, -eta, bdir, eta*beta);
    %}
    
    %{
    % Batch conj.grad.desc. with line search
    [~, Y] = mlpforward(X, W, b);
    [dEdW, dEdb] = mlpbackward(X, t, W, b, Y);
    beta = cgdbeta(dEdW, prevdEdW, dEdb, prevdEdb);
    [~, Wdir] = adjustby(W, dEdW, -1, Wdir, beta);
    [~, bdir] = adjustby(b, dEdb, -1, bdir, beta);
    alpha = cgdlinesearch(X, t, W, b, Wdir, bdir, eta, tolerance);
    [W, Wdir] = adjustby(W, Wdir, alpha);
    [b, bdir] = adjustby(b, bdir, alpha);
    %}
    
    if mlpcheckconvergence(W, b, prevW, prevb, tolerance)
        nconv = nconv + 1;
        if nconv >= convergencecount
            break;
        end
    else
        nconv = 0;
    end
end

if nargout > 2
    y = mlpforward(X, W, b);
    trainerror = mean((y > 0) ~= (t > 0));
end


end