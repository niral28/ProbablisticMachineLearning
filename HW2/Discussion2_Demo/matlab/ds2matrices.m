function [X, Y, Xtest, Ytest] = ds2matrices(ds, p)
%
% [X, Y, Xtest, Ytest] = ds2matrices(ds)
%
% This function partitions a dataset ds to a training input matrix X,
% a training response vector Y, a test input matrix Xtest, and
% a test response vector Y, where p proportion of ds is randomly
% assigned to the training set.
%
% Input:
% ds    - a dataset (.mat), where the last column is the response column
% p     - the proportion of the dataset for training (0 <= p <= 1)
%
% Outputs:
% X     - training input matrix
% Y     - training response vector
% Xtest - test input matrix
% Ytest - test response vector
%

C = dataset2cell(ds);
D = cell2mat(C(2:end,:));
[m, n] = size(D);
trainingSize = floor(m*p);

% ... partition the data matrix D
P = randperm(m);
trainingSet = D(P(1:trainingSize), :);
testSet = D(P(trainingSize+1:end), :);

% ... create the desired matrices
X = trainingSet(:, 1:n-1);
Y = trainingSet(:, n);
Y = logical(Y);
Xtest = testSet(:, 1:n-1);
Ytest = testSet(:, n);
Ytest = logical(Ytest);

return

%% Program Log
%  first created: Chaofan Chen, August 21, 2016
%  last modified:
%  Duke University, Machine Learning, Fall 2016