% Assumes parameters from nnwrapper are already set, and data is already loaded!

[W, b, i, trainerr] = mlplearn(Xtrain, ctrain, numhiddens, etamlp, maxiter);
if trainerr > 0.49 % occasionally, it somehow picks a weird set of weights that lead to quick convergence to nonsensical weights (chance level performance); this loops back in case this happens
    if printoutput
        fprintf('(optimisation failed; retrying...)\n');
    end
    mlpwrapper
    return;
end
if printoutput
    fprintf('MLP training error %.2f%% (%i iterations)\n', 100*trainerr, i);
end

testerr = mean((mlpforward(Xtest, W, b) > 0) ~= (ctest > 0));
if printoutput
    fprintf('MLP test error %.2f%%\n', 100*testerr);
end