% Assumes parameters from nnwrapper are already set, and data is already loaded!

[w, i, trainerr] = perceptronlearn(Xtrain, ctrain, etaperc, maxiter);
if printoutput
    fprintf('Perceptron training error %.2f%% (%i iterations)\n', 100*trainerr, i);
end

testerr = mean(perceptronforward(Xtest, w) ~= ctest);
if printoutput
    fprintf('Perceptron test error %.2f%%\n', 100*testerr);
end