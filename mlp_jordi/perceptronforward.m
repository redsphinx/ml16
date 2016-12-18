function y = perceptronforward(X, w)
%PERCEPTRONFORWARD Calculates the output of a perceptron
% X is an NxM matrix where N is the number of samples, and M the 
% number of input dimensions (plus 1). w is an 1xM vector.
% y is an Nx1 vector (values -1, 1).

y = 2*(X*w' > 0) - 1;


end

