clear 
clc
clf
num_classes = 10;
N = 28*28;
load('mnistAll.mat');

w_per_class = {num_classes};
theta_per_class = {num_classes};
% for i = 0:9
i = 0;
training_data = mnist.train_images(:, :, mnist.train_labels == i);
training_data = double(reshape(training_data, N, size(training_data, 3)) > 0);
m = sum(training_data, 2)/size(training_data, 2);
clamped_state_coupling_expectations = training_data * training_data' / size(training_data,2);
C = clamped_state_coupling_expectations - m * m';
% inverse of C is becoming infs, we are doing something wrong here.
w = eye(N) .* repmat((1 - m.^2), 1, N) - inv(C);
theta = atanh(m) - w*m;
F = (-0.5 * m' * w * m) - (m' * theta) + 0.5 * ...
sum((1 + m) .* log(0.5*(1+m)) + (1 - m) .* log(0.5*(1-m)));
% end

