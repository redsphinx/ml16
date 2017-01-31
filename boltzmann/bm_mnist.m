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

num_classes = 10;
N = 28*28;
load('mnistAll.mat');

    
% testing different noise levels
for noise_level = 0:0.01:0.2
    %% weight calculation
    % weights per class
    w = zeros(N,N,num_classes);
    theta = zeros(N,num_classes);
    % F per class
    F = zeros(1, num_classes);
    for i = 1:10
        % get data belonging to class i
        training_data = mnist.train_images(:, :, mnist.train_labels == i-1);
        % add noise
        training_data = double(training_data) + double(rand(size(training_data)) <= noise_level);
        % threshold and binarize
        training_data = double(reshape(training_data, N, size(training_data, 3)) > 0);
        % mean of training data or clamped statistics for class i
        m = sum(training_data, 2)/size(training_data, 2);
        clamped_state_coupling_expectations = training_data * training_data' / size(training_data,2);
        C = clamped_state_coupling_expectations - m * m';

        % Calculate the parameters per class.
        % C is a singular matrix, so inv(C) is invalid. That's why we're using
        % pinv here.
        w(:,:,i) = eye(N) .* repmat((1 - m.^2), 1, N) - pinv(C);
        theta(:,i) = atanh(m) - w(:,:,i) * m;
        F(i) = (-0.5 * m' * w(:,:,i) * m) - (m' * theta(:,i)) + 0.5 * ...
            sum((1 + m) .* log(0.5 * (1 + m)) + (1 - m) .* log(0.5 * (1 - m)));
    end
    %% Classification
    % Running the classification only for first 500 results.
    test_images = mnist.test_images(:,:,1:500);
    test_labels = mnist.test_labels(1:500);
    test_image_count = size(test_images, 3);
    results = zeros(test_image_count,1);
    w_flat = reshape(w, N*N, 10);

    parfor a = 1:test_image_count
        test_image = test_images(:,:,a);
        test_label = test_labels(a);
        test_image = double(test_image) + double(rand(size(test_image)) <= noise_level);
        binarized_image = double(test_image > 0);
        s = reshape(binarized_image, N, 1);
        logp_per_class = F + reshape(s*s', 1, N*N) * w_flat  + s' * theta;
        [logp, class] = max(logp_per_class);
        results(a) = class - 1;
    end
    %% Accuracy Calculation
    %
    per_class_accuracies = zeros(1,num_classes);
    for digit = 0:9
        class_indices = test_labels == digit;
        per_class_accuracies(digit+1) = mean(results(class_indices, :) == test_labels(class_indices, :));
    end
    noise_level
    mean(results == test_labels)
    per_class_accuracies
end 