function [y, Y] = mlpforward( X, W, b )
%MLPFORWARD Calculates the output of a multi-layer perceptron
% X is an NxM matrix where N is the number of samples, and M the 
% number of input dimensions. W is a cell array of length K,
% where K is the number of layers, and where one cell contains a
% matrix of size MixMo, where Mi is the number of input dimensions of this
% layer (Mi{1} = M), and Mo the number of units in this layer (Mo{i} =
% Mi{i+1}). b is the bias cell array of length K with vector cells length 1xMo{k}
% y is a matrix of size NxMo{K} containing the whole network's outputs and
% Y is a cell array of length K containing each layer's outputs (as
% matrices of size NxMo{k})

numlayers = length(W);
N = size(X, 1);
Y = cell(1, numlayers);
Y{1} = act(repmat(b{1}, [N 1]) + X * W{1});

for k=2:numlayers
    Y{k} = act(repmat(b{k}, [N 1]) + Y{k-1} * W{k});
end

y = Y{numlayers};


end

