clear
close all

% Parameter settings to traverse
noises = [0.01 0.1 0.25 0.5];
testnoises = [false true];
testnoisestrs = {'', ' on both'};

for itestnoise=1:length(testnoises)
    for inoise=1:length(noises)
        noise = noises(inoise);
        testnoise = testnoises(itestnoise);
        
        % Load data
        [Xtrain, ctrain, Xtest, ctest] = loaddata(noise, testnoise);
        
        W = cell(1,10);
        theta = cell(1,10);
        F = single(zeros(1,10));

        numtrain = length(ctrain);
        ptrain = zeros(numtrain, 10);
        numtest = length(ctest);
        ptest = zeros(numtest, 10);

        for k=1:10
            fprintf('Training Boltzmann machine for class %i\n', k-1);

            % Train MFE-LRA Boltzmann machine
            [sic, sisjc] = boltzmannclamped(Xtrain{k}); % N.B. mi = sic
            Cinv = inv(sisjc - sic * sic');
            W{k} = diag(1 ./ (1 - sic .^ 2)) - Cinv;
            theta{k} = atanh(sic) - W{k} * sic;

            % Compute the free energy
            F(k) = sic' * W{k} * sic / 2 + sic' * theta{k} + ...
                   ((1 + sic') * log((1 + sic) / 2) + (1 - sic') * log((1 - sic) / 2)) / 2;

            % Compute log of classification probabilities
            for c=1:10
                ptrain(ctrain == c-1, k) = -F(k) + diag(Xtrain{c} * W{k} * Xtrain{c}') / 2 + ...
                                         Xtrain{c} * theta{k};
                ptest(ctest == c-1, k) = -F(k) + diag(Xtest{c} * W{k} * Xtest{c}') / 2 + ...
                                         Xtest{c} * theta{k};
            end
        end

        % Classify
        [~, trainclassifs] = max(ptrain, [], 2);
        [~, testclassifs] = max(ptest, [], 2);

        % Compute %errors
        trainerr = mean(trainclassifs ~= ctrain+1);
        testerr = mean(testclassifs ~= ctest+1);

        % Compute confusion matrix
        testtargets = zeros(10, numtest);
        testtargets(sub2ind(size(testtargets), ctest'+1, 1:numtest)) = 1;
        [~, testconf] = confusion(testtargets, ptest');
        
        % Display confusion matrix in subplots
        subplot(length(testnoises), length(noises), inoise + (itestnoise-1)*length(noises));
        imagesc(testconf);
        title(sprintf('Noise %i%%%s (train err. %.2f%%, test err. %.2f%%)', 100*noise, testnoisestrs{itestnoise}, 100*trainerr, 100*testerr));
        drawnow
        
        fprintf('Training error: %.2f%%\n', 100*trainerr);
        fprintf('Test error: %.2f%%\n', 100*testerr);
    end
end

suptitle('Confusion matrices on test data for different noise and test noise settings');