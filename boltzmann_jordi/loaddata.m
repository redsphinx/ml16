function [Xtrain, ctrain, Xtest, ctest] = loaddata(noise, usetestnoise)
% LOADDATA Loads the MNIST data as two cell arrays of images per digit.
% Adds given amount of pixel noise to the training data,
% and if requested also to the test data (optional boolean argument).
% Note the pictures are represented as binary matrices with values -1, 1

plotexamples = false;

load mnistAll.mat

if ~exist('usetestnoise', 'var') || ~usetestnoise
    testnoise = 0;
else
    testnoise = noise;
end

% Note that the lecture notes mentioned to only test on 500 test patterns (for an unclear reason)
[Xtrain, ctrain] = loaddataset(mnist.train_images, mnist.train_labels, noise, plotexamples);
[Xtest, ctest] = loaddataset(mnist.test_images(:,:,1:500), mnist.test_labels(1:500), testnoise, false);
	
end


function [Xout, c] = loaddataset(X, c, noise, plotexamples)

len = numel(X(:,:,1));
num = size(X, 3);

X = 2*reshape(single(X > 0), len, num)' - 1;
X = X .* (1-2*(rand(size(X)) < noise)); % taken from course website

if plotexamples
    close all
end

Xout = cell(1,10);
for k=1:10
    Xout{k} = X(c==k-1, :);
    if plotexamples
        figure(k+1)
        for i=find(c==k-1, 4)
            subplot(2,2,i)
            imagesc(reshape(X(i,:), sqrt(len), sqrt(len)));
            colormap gray
        end
    end
end

end