% clear stuff
clear
clc
rng(42)
% sample plot distribution
sigma_square = eye(2) * 0.4;
mu = [0 0];
x1 = -2:0.1:2;
x2 = x1;
[X1, X2] = meshgrid(x1, x2);
X = horzcat(X1(:), X2(:));
T = 3 * mvnpdf([X1(:) X2(:)], mu, sigma_square); 
surf(X1, X2, reshape(T, length(x1), length(x2)))
xlabel('x1')
ylabel('x2')
title('Isotropic 2D Gaussian')