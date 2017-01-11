function [Xtrain, ctrain, Xtest, ctest] = loaddata()
% LOADDATA Returns the MNIST data set as preprocessed for (multilayer) perceptrons

load mnistAll.mat
[Xtrain, ctrain] = preprocess(mnist.train_images, mnist.train_labels);
[Xtest, ctest] = preprocess(mnist.test_images, mnist.test_labels);

end

function [X, c] = preprocess(X, y)

% Select 3s and 7s, convert from uint8 to double
X3 = double(X(:,:,y==3) > 0);
X7 = double(X(:,:,y==7) > 0);

n = size(X3,1) ^ 2;

% Convert images into vectors, add biases
x3 = reshape(X3,n,size(X3,3));	
x7 = reshape(X7,n,size(X7,3));
x3(n+1,:) = 1;
x7(n+1,:) = 1;

X = [x3,x7]';

% Get classes
c = [-ones(size(x3, 2), 1); ones(size(x7, 2), 1)];

end