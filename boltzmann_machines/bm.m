N = 10; % number of neurons
K = 200; % learning steps
T = 500; % number of training states
theta = rand(1,N);
w = rand(N,N);
E = @(s) (-0.5*(sum(w*s)+theta*s));
Z = @(dataset) sum(exp(-E(dataset)));
p_s = @(s,z) exp(-E(s))/z; 
training_dataset = randi([0 1], N,T); % NxT

clamped_state_expectations = sum(training_dataset')/T; % 1xN
clamped_state_expectation_correlations = training_dataset * training_dataset' / T;
for learning_step = 1:K
    z = Z(training_dataset); % 1x1
    p = p_s(training_dataset,z); % 1xT
    free_state_expectations = training_dataset * p'; 
    free_state_expectation_correlations = (training_dataset .* repmat(p, N, 1)) * training_dataset'; 
end