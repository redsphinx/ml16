function [ s ] = sigmf( x )
    s = 1 ./ (1 + exp(-x));
end

