clc
clear
[X, T] = create_training_data(1, 8);
X = reshape(X, size(X,1)^2, size(X,3))';
X = X>0; % treshold X
[matching_table, T] = ohe(T); 
% gd    = vanilla / batch gradient descent
% sgd   = stochastic gradient descent
% mbgd  = mini-batch gradient descent
% cgd   = conjugated gradient descent
LEARNING_METHODS = {'gd', 'sgd', 'mbgd', 'cgd'};
LEARNING_METHOD = 1;
USE_MOMENTUM = 0; % 1=momentum is used, 0=momentum is not used
momentum_parameter = 0.5;

mini_batch_size = 64;

if LEARNING_METHOD == 1
	mini_batch_size = length(X);
elseif LEARNING_METHOD == 2
    mini_batch_size = 1;
elseif LEARNING_METHOD == 3
    mini_batch_size = 64;
elseif LEARNING_METHOD == 4
    mini_batch_size = 1;
end

% make things fit the mini batch size so that we don't get weird results.
X=X(1:length(X) - mod(length(X),mini_batch_size),:);
T=T(1:length(X),:);

% network architecture parameters
D = size(X,2); % nodes in input layer
O = size(T,2); % nodes in output layer. we have 2 classes
L = 4; % number of hidden layers (so not input and not output)
M = 64; % nodes in a hidden layer

%initialize Weights, Bias, predictions Y and activations Z
[Weights, Bias, Y, Z] = initialize_matrices(D, O, L, M, X);

eta = 0.0001; % learning rate
number_of_epochs = 30;

mean_error_per_epoch = zeros(number_of_epochs, 1);
for epoch = 1:number_of_epochs
    epoch
    total_error = zeros(length(X),1);
    Delta_Weights = cell(L+1, 1); % store the weight differences
    Delta_Bias = cell(L+1, 1); % store the bias differences
    % initialize with 0
    for layer=1:L+1
        Delta_Weights{layer} = Weights{layer} * 0;
        Delta_Bias{layer} = Bias{layer} * 0;
    end
    Deltas = cell(L+1,1); % store the errors
    shuffle = randperm(length(X)); % shuffle data
    X = X(shuffle,:);
    T = T(shuffle,:);
    for i = 1:mini_batch_size:length(X)
        mini_batch_range = i:i+mini_batch_size-1;
        % -----------------------------------------------------------------
        % forward pass
        % -----------------------------------------------------------------
        x = X(mini_batch_range,:);
        Z{1} = tanh(x * Weights{1} + repmat(Bias{1}',mini_batch_size, 1));
        for j=2:L+1
            Z{j} = tanh(Z{j-1} * Weights{j} + repmat(Bias{j}',mini_batch_size, 1)) ;
        end
        Y(mini_batch_range,:) = Z{end};
        % -----------------------------------------------------------------
        % backpropagation
        % -----------------------------------------------------------------
        Deltas{end} = mean(Y(mini_batch_range,:) - T(mini_batch_range,:), 1); % delta_k
        [class_value, index] = max(T(mini_batch_range,:)');
        [max_logit, prediction] = max(Y(mini_batch_range,:)');
        total_error(mini_batch_range) = prediction ~= index;
        for j=L:-1:1
            Deltas{j} = (1 - mean(Z{j}, 1).^2) .* (Weights{j+1} * Deltas{j+1}')';
        end
        % -----------------------------------------------------------------
        % updating weights and biases
        % -----------------------------------------------------------------
        if USE_MOMENTUM
            Weights{1} = Weights{1} - (mean(x,1)'* Deltas{1} * eta) - ...
                         momentum_parameter * Delta_Weights{1};
            Delta_Weights{1} = mean(x,1)'* Deltas{1} * eta;
            Bias{1} = Bias{1} - eta * Deltas{1}' - momentum_parameter * ...
                      Delta_Bias{1};
            Delta_Bias{1} = eta * Deltas{1}';
            for j=2:L+1
                Weights{j} = Weights{j} - (eta * Deltas{j}' * mean(Z{j-1}, 1))';
                Delta_Weights{j} = mean(x,1)'* Deltas{j} * eta;
                Bias{j} = Bias{j} - eta * Deltas{j}';
                Delta_Bias{j} = eta * Deltas{j}';
            end 
        else
            Weights{1} = Weights{1} - (mean(x,1)'* Deltas{1} * eta);
            Bias{1} = Bias{1} - eta * Deltas{1}';
            for j=2:L+1
                Weights{j} = Weights{j} - (eta * Deltas{j}' * mean(Z{j-1}, 1))';
                Bias{j} = Bias{j} - eta * Deltas{j}';
            end 
        end           
    end
    mean_error_per_epoch(epoch) = mean(total_error); % accuracy
end

plot(1:number_of_epochs, abs(mean_error_per_epoch))
axis square
xlabel('epoch')
ylabel('error')
title(strcat('Learning Method=',LEARNING_METHODS(LEARNING_METHOD), ', Momentum=', num2str(USE_MOMENTUM)))