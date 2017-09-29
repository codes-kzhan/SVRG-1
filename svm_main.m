%function main(dataset, gridNum)
dataset = 'avazu';
gridNum = 1;
% dataset : toy, covtype, rcv1, avazu, MNIST.

if strcmp(dataset, 'covtype')
    passes = 20;
    factorNR = 0.25;
    factor = 0.25;
    alpha = 0.5;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'rcv1')
    passes = 40;
    factorNR = 1;
    factor = 0.4;
    alpha = 0.01;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'MNIST')
    passes = 50;
    factorNR = 0.5;
    factor = 0.075;
    alpha = 1;
    lambda = 1e-4;
    batchSize = 1;
elseif strcmp(dataset, 'avazu')
    passes = 20;
    factor = 1/4;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 64;
elseif strcmp(dataset, 'criteo')
    passes = 20;
    factor = 1/2;
    lambda = 1e-5;
    batchSize = 64;
end
%% preliminaries
% [Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset
[d, n] = size(Xtrain);
Z = -ytrain' .* Xtrain;
ZT = Z';

% L = max(sum(Xtrain.^2, 1)) * 2 + lambda; % if we did not normalize
L = 2 * sum(Xtrain(:, 1).^2) + lambda; % if we have normalized
mu = lambda;
svmCost = SVM(lambda, L, mu);

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_svm';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = svm_FindOptSolution(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes*20, factor, batchSize);
    save(filename, 'wOpt'); %@TODO
else
    load(filename, 'wOpt');
end
svmCost.optSolution = wOpt;
svmCost.optCost = svmCost.Cost(wOpt, ZT)

%% have fun

% svm_KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum);
% svm_SVRGNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum);
% KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor);
svm_SVRG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% Katyusha(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);