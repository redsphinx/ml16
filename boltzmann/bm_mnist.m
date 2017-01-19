% Write a computer program to implement the Boltzmann machine learning rule 
% as given on pg. 44 of chapter 2. Use N=10 neurons and generate random binary 
% patterns. Use these data to compute the clamped statistics (x_i x_j)_c and 
% (x_i)_c. Use K=200 learning steps. In each learning step use T=500 steps of 
% sequential stochastic dynamics to compute the free statistics (x_i x_j) and 
% (x_i). Test the convergence by plotting the size of the change in weights 
% versus iteration. A much more efficient learning method can be obtained by 
% using the mean field theory and the linear response correction. Build a 
% classifier for the MNIST data based on the Boltzmann Machine as described 
% in 2.5.1
%% preparation 
clear 
clc
clf
num_classes = 10;
N = 28*28;
load('mnistAll.mat');
%% weight calculation
w = zeros(N,N,num_classes);
theta = zeros(N,num_classes);
F = zeros(1, num_classes);
for i = 1:10
    % get data belonging to class i
    training_data = mnist.train_images(:, :, mnist.train_labels == i-1);
    % thresholding
    training_data = double(reshape(training_data, N, size(training_data, 3)) > 0);
    % mean of training data or clamped statistics for class i
    m = sum(training_data, 2)/size(training_data, 2);
    clamped_state_coupling_expectations = training_data * training_data' / size(training_data,2);
    C = clamped_state_coupling_expectations - m * m';

    % C is a singular matrix, so inv(C) is invalid. That's why we're using
    % pinv here.
    w(:,:,i) = eye(N) .* repmat((1 - m.^2), 1, N) - pinv(C);
    theta(:,i) = atanh(m) - w(:,:,i) * m;
    F(i) = (-0.5 * m' * w(:,:,i) * m) - (m' * theta(:,i)) + 0.5 * ...
        sum((1 + m) .* log(0.5 * (1 + m)) + (1 - m) .* log(0.5 * (1 - m)));
end
%% Classification
start = now;
test_image_count = size(mnist.test_images, 3);
results = zeros(test_image_count,1);
w_flat = reshape(w, N*N, 10);
parfor a = 1:test_image_count
    test_image = mnist.test_images(:,:,a);
    test_label = mnist.test_labels(a);
    binarized_image = double(test_image > 0);
    s = reshape(binarized_image, N, 1);
    logp_per_class = F + reshape(s*s', 1, N*N) * w_flat  + s' * theta;
    [logp, class] = max(logp_per_class);
    results(a) = class - 1;
end
now - start
%%
% accuracy = 0.7574
sum(results == mnist.test_labels)/test_image_count

