function converged = mlpcheckconvergence(W, b, prevW, prevb, tolerance)
% MLPCHECKCONVERGENCE Checks whether learning has converged, given some tolerance.
% Given the current set of weights W and biases b, and the ones from the
% previous learning iteration prevW and prevb, checks whether learning has
% converged by checking the squared Euclidean distance against a given tolerance.

K = length(W);
abschange = 0;

for k=1:K
    deltaWsq = (W{k} - prevW{k}) .^ 2;
    abschange = abschange + sum(deltaWsq(:)) + sum((b{k} - prevb{k}) .^ 2);
end

converged = abschange < tolerance;


end