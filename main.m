%function main(dataset, gridNum)
dataset = 'criteo';
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
    factor = 5;
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

L = max(sum(Xtrain.^2, 1)) / 4 + lambda;
mu = lambda;
logCost = ObjFunc(lambda, L, mu);

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_logistic';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = FindOptSolution(logCost, Xtrain, ytrain, Xtest, ytest, 14, factor, batchSize);
    save(filename, 'wOpt');
else
    load(filename, 'wOpt');
end
logCost.optSolution = wOpt;
logCost.optCost = logCost.Cost(wOpt, Xtrain, ytrain)

%% have fun

% SVRGNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% KatyushaNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SVRG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% Katyusha(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
box on
grid on
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
