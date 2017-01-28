clc
clear
dummydata
% TODO: import data for actual X and labels T
% TODO: change code to handle more than 1 layer
% TODO: change code to handle different learning methods
% TODO: change output such that it classifies, so add sigmoid

clearvars -except X T X1 X2 x1 x2
% implement neural net
D = 2; % nodes in input layer
O = 1; % nodes in output layer
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
Bias = rand(L+1, 1) - 0.5;

% initial approximation with random weights
Y = zeros(length(X),1); % store our predections for each datapoint
Z = cell(L+1, 1); % tanh(activation)

for i = 1:length(X)
    x = X(i,:);
    Z{1} = tanh(x * Weights{1} + Bias(1));
    for j=2:L+1
        Z{j} = tanh(Z{j-1} * Weights{j}) + Bias(j);
    end
    if O == 1
        Y(i) = Z{end};  
    else
        Y(i) = sigmf(Z{end}); %  sigmoid to create classification predictions when we have 2 classes or more
    end
end
% surf(X1, X2, reshape(Y, length(x1), length(x2)))
% xlabel('x1')
% ylabel('x2')
% title(sprintf('Neural Network at epoch 0'))


eta = 0.001; % learning rate
number_of_epochs = 100;
interval = 50;
number_of_plots = number_of_epochs / interval;
plot_counter = 0;

for epoch = 1:number_of_epochs
    epoch
    Deltas = cell(L+1,1); % store the errors
    for i = 1:length(X)
        % forward pass
        x = X(i,:);
        Z{1} = tanh(x * Weights{1} + Bias(1));
        for j=2:L+1
            Z{j} = tanh(Z{j-1} * Weights{j}) + Bias(j);
        end
        Y(i) = Z{end};

        % backpropagation
        Deltas{1} = Y(i) - T(i); % delta_k
        ks = 2:L+1;
        ks = flip(ks);
        for j=2:L+1
            k = ks(j-1);
            Deltas{j} = (1 - Z{k}.^2)' .* ( (Weights{k} * Deltas{j-1}) ); % fixed
        end
        Deltas = flip(Deltas);
        
        Weights{1} = Weights{1} - (eta * Deltas{1} * x)'; % fixed
        for j=2:L+1
            Weights{j} = Weights{j} - eta * (Deltas{j} * Z{j-1})';
        end
        
    end
    if(mod(epoch,interval) == 0)
        plot_counter = plot_counter + 1;
%         sprintf('plot')
%         figure;
%         surf(X1, X2, reshape(Yhat, length(x1), length(x2)))
%         title(sprintf('epoch: %d',epoch))

        subplot(2,number_of_plots/2,plot_counter)
        surf(X1, X2, reshape(Y, length(x1), length(x2)))
        axis square
        title(sprintf('epoch: %d',epoch))
    end
end