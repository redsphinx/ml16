clc
clear
dummydata
% TODO: import data for actual X and labels T
% TODO: change code to handle more than 1 layer
% TODO: change code to handle different learning methods
% TODO: change output such that it classifies, so add sigmoid
%%
clearvars -except X T X1 X2 x1 x2
% implement neural net
D = 2; % nodes in input layer
O = 1; % nodes in output layer
L = 2; % number of hidden layers (so not input and not output)
M = 8; % nodes in a hidden layer

% we have L + 1 weight matrices 
Weights = cell(L+1, 1);
for i = 1:L+1
    if i ~= 1 && i ~= L+1
        Weights{i} = create_weight_matrix(M+1, M+1);
    elseif i == 1
        % bias implied in the "+1"
        Weights{i} = create_weight_matrix(D+1, M+1);
    elseif i == L+1
        % bias implied in the "+1"
        Weights{i} = create_weight_matrix(M+1, O);
    end
end
% TODO: remove bias component later
%%
% initial approximation with random weights
Y = zeros(length(X),1); % store our predections for each datapoint
Z = cell(L+1, 1); % tanh(activation)

for i = 1:length(X)
    x = [X(i,:) 1];
%     Z{1} = [tanh(x * Weights{i}) 1];
    Z{1} = tanh(x * Weights{1});
    for j=2:L+1
%         Z{j} = [tanh(Z{i} * Weights{j}) 1];
        Z{j} = tanh(Z{j-1} * Weights{j});
    end
    Y(i) = Z{end}(1); % remove the bias term
end
surf(X1, X2, reshape(Y, length(x1), length(x2)))
xlabel('x1')
ylabel('x2')
title(sprintf('Neural Network at epoch 0'))
%%
% A4E2_3
number_of_epochs = 20;
interval = 2;
number_of_plots = number_of_epochs / interval;
plot_counter = 0;
% learning rate
eta = 0.1;

%to plot the initial approximation with random weights
subplot(2,number_of_plots/2,1)
surf(X1, X2, reshape(Y, length(x1), length(x2)))
axis square
title(sprintf('epoch 0'))
Y = zeros(length(X),1);

for epoch = 1:number_of_epochs
    epoch
    for i = 1:length(X)
        x = [X(i,:) 1];
        z = [tanh(x * W1) 1];
        y = z * W2;
        Y(i) = y;
        % backpropagation
        delta_k = y - T(i);
        delta_j = (1 - z.^2) .* (W2' * delta_k); % (1x(M+1))
        delta_j = delta_j(1:M); %remove the last "bias" item
        % (1 - z^2): da/dW1
        dEdW1 = delta_j' * x;
        dEdW2 = delta_k * z';

        W1 = W1 - eta * dEdW1';
        W2 = W2 - eta * dEdW2;
    end
    if(mod(epoch,interval) == 0)
        plot_counter = plot_counter + 1;
%         sprintf('plot')
%         figure;
%         surf(X1, X2, reshape(Yhat, length(x1), length(x2)))
%         title(sprintf('epoch: %d',epoch))

        subplot(2,number_of_plots/2,plot_counter+1)
        surf(X1, X2, reshape(Y, length(x1), length(x2)))
        axis square
        title(sprintf('epoch: %d',epoch))
    end
end