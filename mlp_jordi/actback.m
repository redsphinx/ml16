function diff = actback(Y)
% ACTBACK Returns this layer's derivative w.r.t. its weights given the layer's output
% tanh' = 1-tanh^2

diff = 1 - Y.^2;


end

