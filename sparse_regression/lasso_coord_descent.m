function betas = lasso_coord_descent(xtrain, ytrain, gammas, T, tolerance)

% Convert data to more convenient form
S = @(arg,gamma) sign(arg) .* max(0, abs(arg) - gamma);
[n,p] = size(xtrain);
chi = xtrain * xtrain' / p;
b = xtrain * ytrain' / p;
betas = NaN(numel(gammas), n);
beta = zeros(n,1);

% Cycle through gammas (hot start)
for igamma=1:numel(gammas)
    gamma = gammas(igamma);
    t = 0;
    prevbeta = NaN(n,1);

    % Main loop
    while ~(norm(beta-prevbeta) < tolerance || t >= T)
        t = t + 1;
        prevbeta = beta;

        for j=1:n
            beta(j) = S(b(j) - chi(j,:) * beta + chi(j,j) * beta(j), gamma);
        end
    end
    
    betas(igamma,:) = beta;
end

end