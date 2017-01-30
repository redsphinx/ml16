clc
clear
[X, T] = create_training_data(1, 8);
[X_test, T_test] = create_testing_data(1, 8);
X = reshape(X, size(X,1)^2, size(X,3))';
X_test = reshape(X_test, size(X_test,1)^2, size(X_test,3))';
X = X>0; % treshold X
X_test = X_test>0; % treshold X
[matching_table, T] = ohe(T); 
[matching_table, T_test] = ohe(T_test); 
% gd    = vanilla / batch gradient descent
% sgd   = stochastic gradient descent
% mbgd  = mini-batch gradient descent
% cgd   = conjugated gradient descent

training_results_learning_methods_per_epoch = zeros(3, 100);
testing_results_learning_methods_per_epoch = zeros(3, 100);

for method = 1:3
    LEARNING_METHODS = {'gd', 'sgd', 'mbgd', 'cgd'};
    LEARNING_METHOD = method;
    USE_MOMENTUM = 0; % 1=momentum is used, 0=momentum is not used
    momentum_parameter = 0.5;

    mini_batch_size = 64;

    if LEARNING_METHOD == 1
        mini_batch_size = length(X);
    elseif LEARNING_METHOD == 2
        mini_batch_size = 32;
%         mini_batch_size = 1;
    elseif LEARNING_METHOD == 3
        mini_batch_size = 64;
    elseif LEARNING_METHOD == 4
        mini_batch_size = 1;
    end

    % make things fit the mini batch size so that we don't get weird results.
    X=X(1:length(X) - mod(length(X),mini_batch_size),:);
    T=T(1:length(X),:);
%     X_test=X_test(1:length(X_test) - mod(length(X_test),mini_batch_size),:);
%     T_test=T_test(1:length(T_test),:);

    % network architecture parameters
    D = size(X,2); % nodes in input layer
    O = size(T,2); % nodes in output layer. we have 2 classes
    L = 6; % number of hidden layers (so not input and not output)
    M = 256; % nodes in a hidden layer

    %initialize Weights, Bias, predictions Y and activations Z
    [Weights, Bias, Y, Z] = initialize_matrices(D, O, L, M, X);
    Y_test = zeros(length(X_test), O);

    eta = 0.0001; % learning rate
    number_of_epochs = 100;

    mean_error_per_epoch = zeros(number_of_epochs, 1);
    for epoch = 1:number_of_epochs
        epoch
        total_error = zeros(length(X),1);
        Delta_Weights = cell(L+1, 1); % store the weight differences
        Delta_Bias = cell(L+1, 1); % store the bias differences
        Deltas = cell(L+1,1); % store the errors
        old_Deltas = cell(L+1,1); % store the previous Deltas
        % initialize with 0
        for layer=1:L+1
            Delta_Weights{layer} = Weights{layer} * 0;
            Delta_Bias{layer} = Bias{layer} * 0;
    %         old_Deltas{layer} = Deltas{layer} * 0;
        end
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
            if LEARNING_METHOD == 4
                if i==1
                    for layer=1:L+1
                       old_Deltas{layer} = Deltas{layer} * 0;
                    end
                end
                for j = 1:L+1
                    lambda = 1; % don't know how to compute lambda
                    d0 = old_Deltas{j};
                    Beta = ((Deltas{j} - old_Deltas{j}).*Deltas{j}) / norm(d0);
                    d1 = Deltas{j} + Beta.*d0;
                    Weights{j} = Weights{j} + repmat(lambda*d1,size(Weights{j}, 1), 1);
                    Bias{j} = Bias{j} + (lambda*d1)';
                end
                old_Deltas = Deltas;
            else
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
        end
        mean_error_per_epoch(epoch) = mean(total_error);
        % calculate test error
        Z_test = cell(L+1, 1); % tanh(activation)

        for test_data=1:length(X_test)
            x_test = X_test(test_data,:);
            Z_test{1} = tanh(x_test * Weights{1} + Bias{1}');
            for j=2:L+1
                Z_test{j} = tanh(Z_test{j-1} * Weights{j} + Bias{j}') ;
            end
            Y_test(test_data,:) = Z_test{end};
        end

        training_results_learning_methods_per_epoch(method, epoch) = mean_error_per_epoch(epoch);
        % TODO: fix the way to calculate the test error down below
        testing_results_learning_methods_per_epoch(method, epoch) = mean(Y_test-T_test)
    end 
end