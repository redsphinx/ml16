clear
close all
hold on

% Network parameters
etamlp = 0.001; % learning rate
etaperc = 0.1; % learning rate perceptron
maxiter = 2500; % maximum number of updates
numhiddenstotry = [2 4 8 16 32 64 128 256]; % numbers of hidden units to try

% Simulation parameters
nrepetitions = 25;
results = zeros(nrepetitions, 3);
allresults = cell(1);
legendnames = {'Perceptron'};
printoutput = true;

% Load data
[Xtrain, ctrain, Xtest, ctest] = loaddata();

% Evaluate perceptrons
for irep=1:nrepetitions
    perceptronwrapper
    results(irep,1) = i;
    results(irep,2) = 100*trainerr;
    results(irep,3) = 100*testerr;
end
allresults{1} = results;

figure(1)
hold on
scatter(results(:,1), results(:,2));
scatter(results(:,1), results(:,3));
xlabel('iterations till convergence');
ylabel('Error (%)');
title('Perceptron performance');
legend('train', 'test');

figure(2*length(numhiddenstotry)+2)
scatter(results(:,1), results(:,3));
drawnow

% Evaluate MLPs
for ihidden=1:length(numhiddenstotry)
    numhiddens = numhiddenstotry(ihidden);
    for irep=1:nrepetitions
        mlpwrapper
        results(irep,1) = i;
        results(irep,2) = 100*trainerr;
        results(irep,3) = 100*testerr;
    end
    allresults{ihidden+1} = results;
    
    figure(ihidden+1)
    hold on
    scatter(results(:,1), results(:,2));
    scatter(results(:,1), results(:,3));
    xlabel('iterations till convergence');
    ylabel('Error (%)');
    title(sprintf('MLP performance with %i hidden units', numhiddens));
    legend('Training error', 'Test error');
    
    figure(2*length(numhiddenstotry)+2)
    hold on
    scatter(results(:,1), results(:,3));
    legendnames{end+1} = sprintf('MLP, %i units', numhiddens);
    drawnow
end


% Evaluate MLPs with two hidden layers
for ihidden=1:length(numhiddenstotry)
    numhiddens = [numhiddenstotry(ihidden) numhiddenstotry(ihidden)];
    for irep=1:nrepetitions
        mlpwrapper
        results(irep,1) = i;
        results(irep,2) = 100*trainerr;
        results(irep,3) = 100*testerr;
    end
    allresults{end+1} = results;

    figure(length(numhiddenstotry)+1+ihidden)
    hold on
    scatter(results(:,1), results(:,2));
    scatter(results(:,1), results(:,3));
    xlabel('iterations till convergence');
    ylabel('Error (%)');
    title(sprintf('MLP performance with two hidden layers of %i units', numhiddens(1)));
    legend('train', 'test');

    figure(2*length(numhiddenstotry)+2)
    hold on
    scatter(results(:,1), results(:,3));
    legendnames{end+1} = sprintf('MLP, 2x%i units', numhiddens(1));
    drawnow
end


% Aggregate results and finish aggregated figures
bestresults = cellfun(@(x) min(x(:,3)), allresults);

figure(2*length(numhiddenstotry)+2)
hold on
set(gca,'ColorOrderIndex',1)
means = zeros(1,2*length(numhiddenstotry)+1);
for i=1:2*length(numhiddenstotry)+1
    means(i) = mean(allresults{i}(:,3));
    plot([0 2500], repmat(means(i), [1 2]), '--');
end
xlabel('#iterations till convergence');
ylabel('Test error (%)');
title('Performance of different models');
legend(legendnames, 'Location', 'BestOutside');

figure
semilogx(numhiddenstotry, means(2:length(numhiddenstotry)+1));
hold on
semilogx(numhiddenstotry, repmat(means(1), [1, length(numhiddenstotry)]));
semilogx(numhiddenstotry, means(length(numhiddenstotry)+2:end));
xlabel('#hidden units');
ylabel('Test error (%)');
title('Performance of the MLP for different numbers of hidden units');
legend('MLP', 'Perceptron', 'MLP (2 layers)');

save allresults