%function main(dataset, gridNum)
dataset = 'MNIST';
gridNum = 1;
% dataset : toy, covtype, rcv1, avazu, MNIST.

if strcmp(dataset, 'covtype')
    passes = 20;
    factor = 1/2;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'rcv1')
    passes = 20;
    factor = 3/8;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'MNIST')
    passes = 25;
    factor = 0.1;
    lambda = 1e-4;
    batchSize = 1;
elseif strcmp(dataset, 'avazu')
    passes = 20;
    factor = 1/4;
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
Z = -ytrain' .* Xtrain;
ZT = Z';

% L = max(sum(Xtrain.^2, 1)) / 4 + lambda;
L = 2 * max(svds(Z, 1)^2, svds(Z, 1, 'smallest')) + lambda;
mu = lambda;
svmCost = SVM(lambda, L, mu);

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_svm';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = svm_FindOptSolution(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes*20, factor, batchSize);
    % save(filename, 'wOpt'); %@TODO
else
    load(filename, 'wOpt');
end
svmCost.optSolution = wOpt;
svmCost.optCost = svmCost.Cost(wOpt, ZT)

%% have fun

% SVRGNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor);
% svm_SVRG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% Katyusha(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
box on
grid on
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
