clear
clc
clf

N = 10; % number of neurons
K = 200; % learning steps
T = 500; % number of training states
theta = rand(1,N);
w = rand(N,N);
w(1:N+1:N*N) = 0;

training_dataset = randi([0 1], N,T); % NxT
learning_rate = 0.04;

% calculating clamped statistics
% <s_i>_c
clamped_state_expectations = sum(training_dataset, 2)'/T; % 1xN
% <s_{ij}>_c
clamped_state_correlation_expectations = training_dataset * training_dataset' / T; % NxN

% array to hold convergence data
delta_ws = zeros(1,K);
delta_thetas = zeros(1,K);
for learning_step = 1:K
    E = (-0.5*(sum(w*training_dataset)+theta*training_dataset));
    Z = sum(exp(-E));
    p_s = exp(-E)/Z; 
    % calculating free statistics
    % <s_i>
    free_state_expectations = p_s * training_dataset'; % 1xN
    % <s_{ij}>
    free_state_correlation_expectations = (training_dataset .* repmat(p_s, N, 1)) * training_dataset';
    % delta_w = <s_{ij}>_c - <s_{ij}>
    delta_w = clamped_state_correlation_expectations - free_state_correlation_expectations;
    % set diagonal of delta_w to 0
    delta_w(1:N+1:N*N) = 0;
    
    % delta_theta = <s_i>_c - <s_i>
    delta_theta = clamped_state_expectations - free_state_expectations;
    
    % apply learning
    w = w + learning_rate * delta_w;
    theta = theta + learning_rate * delta_theta;
    
    % store delta values to show convergence
    delta_ws(learning_step) = mean(mean(abs(delta_w)));
    delta_thetas(learning_step) = mean(abs(delta_theta));
end

% hodor
hold on
plot(delta_ws, 'r')
plot(delta_thetas, 'b')
hold off
xlabel('learning steps')
legend('\Delta{w}', '\Delta\theta')