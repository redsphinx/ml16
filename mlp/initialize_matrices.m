function [Weights, Bias, Y, Z] = initialize_matrices(D, O, L, M, X)
    % we have L + 1 weight matrices 
    Weights = cell(L+1, 1);
    for i = 1:L+1
        if i ~= 1 && i ~= L+1
            Weights{i} = create_weight_matrix(M, M);
        elseif i == 1
            Weights{i} = create_weight_matrix(D, M);
        elseif i == L+1
            Weights{i} = create_weight_matrix(M, O);
        end
    end
    % we have L + 1 biases
    Bias = cell(L+1, 1);
    for i = 1:L+1
        if i == L+1
            Bias{i} = rand(O, 1) - 0.5;
        else
            Bias{i} = rand(M, 1) - 0.5;
        end
    end
    Y = zeros(length(X), O); % store our predections for each datapoint
    Z = cell(L+1, 1); % tanh(activation)
end