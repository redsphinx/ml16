clc
clear
dummydata
% TODO: import data for actual X and labels Y
% TODO: change code to handle more than 1 layer
% TODO: change code to handle different learning methods
%%
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
        % bias implied in the "+1"
        Weights{i} = create_weight_matrix(D+1, M);
    elseif i == L+1
        % bias implied in the "+1"
        Weights{i} = create_weight_matrix(M+1, O);
    end
    
end
%%
% learning rate
eta = 0.1;

% initial approximation with random weights
Yhat = zeros(length(X),1);
for i = 1:length(X)
    x = [X(i,:) 1];
    z = [tanh(x * W1) 1];
    y = z * W2;
    Yhat(i) = y;
end
% surf(X1, X2, reshape(Yhat, length(x1), length(x2)))
% xlabel('x1')
% ylabel('x2')
% title(sprintf('Neural Network at epoch 0'))
%%
% A4E2_3
number_of_epochs = 20;
interval = 2;
number_of_plots = number_of_epochs / interval;
plot_counter = 0;

%to plot the initial approximation with random weights
subplot(2,number_of_plots/2,1)
surf(X1, X2, reshape(Yhat, length(x1), length(x2)))
axis square
title(sprintf('epoch 0'))
Yhat = zeros(length(X),1);

for epoch = 1:number_of_epochs
    epoch
    for i = 1:length(X)
        x = [X(i,:) 1];
        z = [tanh(x * W1) 1];
        y = z * W2;
        Yhat(i) = y;
        % backpropagation
        delta_k = y - Y(i);
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
        surf(X1, X2, reshape(Yhat, length(x1), length(x2)))
        axis square
        title(sprintf('epoch: %d',epoch))
    end
end