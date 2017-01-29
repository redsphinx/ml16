clc
clear
[X, T] = create_training_data(1, 8);
X = reshape(X, size(X,1)^2, size(X,3))';
X = X>0;
[matching_table, T] = ohe(T); 
% TODO: change code to handle different learning methods

% implement neural net
D = size(X,2); % nodes in input layer
O = size(T,2); % nodes in output layer. we have 2 classes
L = 5; % number of hidden layers (so not input and not output)
M = 64; % nodes in a hidden layer

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

% % we have L + 1 biases
Bias = cell(L+1, 1);
for i = 1:L+1
    if i == L+1
        Bias{i} = rand(O, 1) - 0.5;
    else
        Bias{i} = rand(M, 1) - 0.5;
    end
end

% initial approximation with random weights
Y = zeros(length(X),O); % store our predections for each datapoint
Z = cell(L+1, 1); % tanh(activation)

eta = 0.1; % learning rate
number_of_epochs = 5;

mean_error_per_epoch = zeros(number_of_epochs, 1);
for epoch = 1:number_of_epochs
    epoch
    total_error = zeros(length(X),1);
    Deltas = cell(L+1,1); % store the errors
    for i = 1:length(X)
        % forward pass
        x = X(i,:);
        Z{1} = tanh(x * Weights{1} + Bias{1}');
        for j=2:L+1
            Z{j} = tanh(Z{j-1} * Weights{j} + Bias{j}') ;
        end
        Y(i,:) = Z{end};
        % softmax the output
%         a = exp(Y(i,1)) / (exp(Y(i,1)) + exp(Y(i,2)));
%         b = exp(Y(i,2)) / (exp(Y(i,1)) + exp(Y(i,2)));
%         Y(i,:) = [a,b];
        
        % backpropagation
        Deltas{end} = Y(i,:) - T(i,:); % delta_k
        [class_value, index] = max(T(i,:));
        [max_logit, prediction] = max(Deltas{end});
        total_error(i) = prediction ~= index;
        for j=L:-1:1
            Deltas{j} = (1 - Z{j}.^2) .* (Weights{j+1} * Deltas{j+1}')';
        end
        
        % update weights and biases
        Weights{1} = Weights{1} - ( x'* Deltas{1} * eta);
        Bias{1} = Bias{1} - eta * Deltas{1}'; %assumption
        for j=2:L+1
            Weights{j} = Weights{j} - (eta * Deltas{j}' * Z{j-1})';
            Bias{j} = Bias{j} - eta * Deltas{j}';
        end
        
    end
    mean_error_per_epoch(epoch) = mean(total_error); % accuracy

end

plot(1:number_of_epochs, abs(mean_error_per_epoch))
axis square
xlabel('epoch')
ylabel('error')