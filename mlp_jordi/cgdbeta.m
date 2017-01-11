function beta = cgdbeta(dEdW, prevdEdW, dEdb, prevdEdb)
% CGDBETA Returns the Polak-Ribiere beta given current and previous gradients.

if ~iscell(prevdEdW) || ~iscell(prevdEdb)
    beta = 0;
    return;
end

K = length(dEdW);
betanum = 0;
betaden = 0;

% Polak-Ribiere
num = @(dE, prevdE) dE' * (dE - prevdE);
den = @(prevdE) prevdE' * prevdE;

for k=1:K
    betanum = betanum + num(dEdW{k}(:), prevdEdW{k}(:)) + num(dEdb{k}(:), prevdEdb{k}(:));
    betaden = betaden + den(prevdEdW{k}(:)) + den(prevdEdb{k}(:));
end

beta = max(0, betanum / betaden);


end