function [dEdW, dEdb] = mlpbackward(X, t, W, b, Y)
%MLPFORWARD Calculates the error gradient for a multi-layer perceptron
% X is an NxM matrix where N is the number of samples, and M the 
% number of input dimensions, and t is a NxMo{K} matrix containing the 
% desired network outputs. W is a cell array of length K,
% where K is the number of layers, and where one cell contains a
% matrix of size MixMo, where Mi is the number of input dimensions of this
% layer (Mi{1} = M), and Mo the number of units in this layer (Mo{i} =
% Mi{i+1}). b is the bias vector of length K.
% Y is a cell array of length K containing each layer's outputs (as
% matrices of size NxMo{k})
% dEdW is a cell array with the same dimensions as W, which contains the
% gradient of E with respect to elements of W, and dEdb is similarly the
% gradient vector w.r.t. b, with the same dimensions as b

numlayers = length(b);
dEdW = cell(1, numlayers);
dEdb = cell(1, numlayers);

delta = (Y{numlayers} - t) .* actback(Y{numlayers});
for k=numlayers:-1:2
    dEdW{k} = Y{k-1}' * delta;
    dEdb{k} = sum(delta, 1);
    delta = actback(Y{k-1}) .* (delta * W{k}');
end

dEdW{1} = X' * delta;
dEdb{1} = sum(delta, 1);



end