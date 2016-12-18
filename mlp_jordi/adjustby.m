function [ar, dir] = adjustby(ar, varargin)
% ADJUSTBY Adjusts all cells in a cell array by the specified amounts and multipliers
% [ar, dir] = cgdadjustby(ar, varargin) takes as input a cell array ar and 
% a number of paired arguments amts, mult. amts is a cell array of 
% the same shape as ar, and mult is a scalar or matrix of the same shape as every amts{k}.
% For each parameter pair (amts, mult) supplied after ar, and for each cell index k,
% accumulates dir{k} = dir{k} + amts{k}.*mult (starting at the zero matrix),
% and after this loop has finished, ar{k} = ar{k} + dir{k} is computed

nvarargs = length(varargin);
if nvarargs  < 2
    error('Not enough parameters');
end
if mod(nvarargs, 2) == 1
    error('Faulty parameter list');
end

K = length(ar);
dir = cell(1, K);

for k=1:K
    dir{k} = varargin{1}{k} .* varargin{2};
    for i=3:2:nvarargs
        if iscell(varargin{i}) % N.B. Certain cell arrays are initialised to the scalar NaN
            dir{k} = dir{k} + varargin{i}{k} .* varargin{i+1};
        end
    end
    ar{k} = ar{k} + dir{k};
end

end
