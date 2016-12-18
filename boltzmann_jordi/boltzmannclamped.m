function [sic, sisjc] = boltzmannclamped(X)
% BOLTZMANNCLAMPED Computes clamped statistics <s_i>_c and <s_is_j>_c

P = size(X, 1);
sic = mean(X, 1)';
sisjc = X' * X / P;


end