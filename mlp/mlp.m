clc
clear
[X, T] = create_training_data(1, 8);
X = reshape(X, size(X,1)^2, size(X,3))';
% create one-hot-encoding for T, with first argument in 
%create_training_data=1 and second argument in create_training_data=0
T = ohe(T, 1, 0); 
% TODO: change code to handle different learning methods
% TODO: change output such that it classifies, so add sigmoid

clearvars -except X T X1 X2 x1 x2
% implement neural net
D = size(X,2); % nodes in input layer
O = 1; % nodes in output layer. we have 2 classes
L = 1; % number of hidden layers (so not input and not output)
M = 8; % nodes in a hidden layer

% we have L + 1 weight matrices 
Weights = cell(L+1, 1);
for i = 1:L+1
    if i ~= 1 && i ~= L+1
        Weights{i} = create_weight_matrix(M, M);
    elseif i == 1
        % no implied bias
        Weights{i} = create_weight_matrix(D, M);
    elseif i == L+1
        % no implied bias
        Weights{i} = create_weight_matrix(M, O);
    end
end

% % we have L + 1 biases
% Bias = rand(L+1, 1) - 0.5;
Bias = cell(L+1, 1);
for i = 1:L+1
    if i == L+1
        Bias{i} = rand() - 0.5;
    else
        Bias{i} = rand(M, 1) - 0.5;
    end
end

% initial approximation with random weights
Y = zeros(length(X),1); % store our predections for each datapoint
Z = cell(L+1, 1); % tanh(activation)

for i = 1:length(X)
    x = X(i,:);
%     Z{1} = tanh(x * Weights{1} + Bias(1));
	Z{1} = tanh(x * Weights{1} + Bias{1}');

    for j=2:L+1
%         Z{j} = tanh(Z{j-1} * Weights{j}) + Bias(j);
        Z{j} = tanh(Z{j-1} * Weights{j}+ Bias{j}') ;
    end
    Y(i) = round(sigmf(Z{end}));
%     if O == 1
%         Y(i) = Z{end};  
%     else
%         Y(i) = sigmf(Z{end}); %  sigmoid to create classification predictions when we have 2 classes or more
%     end
end

eta = 0.01; % learning rate
number_of_epochs = 10;
interval = number_of_epochs / 10;
number_of_plots = number_of_epochs / interval;
plot_counter = 0;

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
%         Y(i) = Z{end};
        Y(i) = round(sigmf(Z{end}));

        % backpropagation
        Deltas{1} = Y(i) - T(i); % delta_k
        total_error(i) = Deltas{1};
        ks = 2:L+1;
        ks = flip(ks);
        for j=2:L+1
            k = ks(j-1);
            Deltas{j} = (1 - Z{k}.^2)' .* ( (Weights{k} * Deltas{j-1}) ); % fixed
%             Deltas{j} = (1 - Z{k-1}.^2)' .* ( (Weights{k} * Deltas{j-1}) );
        end
        Deltas = flip(Deltas);
        
        % update weights and biases
        Weights{1} = Weights{1} - (eta * Deltas{1} * x)'; % fixed
        Bias{1} = Bias{1} - (eta * Deltas{1});
        for j=2:L+1
            Weights{j} = Weights{j} - eta * (Deltas{j} * Z{j-1})';
            Bias{j} = Bias{j} - eta * Deltas{j};
        end
        
    end
    mean_error_per_epoch(epoch) = mean(total_error);
%     % plot error per instance
%     if(mod(epoch,interval) == 0)
%         plot_counter = plot_counter + 1;
%         subplot(2,number_of_plots/2,plot_counter)
% %         surf(X1, X2, reshape(Y, length(x1), length(x2)))
%         plot(1:length(T), total_error')
%         axis square
%         title(sprintf('epoch: %d',epoch))
%     end
end

plot(1:number_of_epochs, mean_error_per_epoch)
axis square
xlabel('epoch')
ylabel('Error')