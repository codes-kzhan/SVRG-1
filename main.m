% function main(dataset, gridNum)
dataset = 'MNIST';
gridNum = 3;
% dataset : toy, covtype, rcv1, avazu, MNIST. HIGGS

if strcmp(dataset, 'covtype')
    passes = 20;
    factor = 1/2;
    factorNR = 1/2;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'rcv1')
    passes = 20;
    factor = 3/8;
    factorA = 3/8;
    factorIAG = 0.05;
    factorNR = 3/8;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'MNIST')
    passes = 20;
    factor = 0.12;
    factorNR = 0.12;
    factorA = 0.1;
    factorIAG = 0.01;
    factorGD = 10;
    alpha = 0.6;
    lambda = 1e-4;
    batchSize = 1;
elseif strcmp(dataset, 'avazu')
    passes = 20;
    factor = 1/4;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 64;
elseif strcmp(dataset, 'criteo')
    passes = 6;
    factor = 1/2;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 64;
elseif strcmp(dataset, 'HIGGS')
    passes = 20;
    factor = 0.1;
    factorNR = 0.1;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 1;
end
%% preliminaries
% [Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset

L = max(sum(Xtrain.^2, 1)) / 4 + lambda;
mu = lambda;
logCost = ObjFunc(lambda, L, mu);

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_logistic_norm';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = FindOptSolution(logCost, Xtrain, ytrain, Xtest, ytest, passes*20, factor, batchSize);
    save(filename, 'wOpt');
else
    load(filename, 'wOpt');
end
logCost.optSolution = wOpt;
logCost.optCost = logCost.Cost(wOpt, Xtrain, ytrain)

%% have fun

filename = strcat('../data/', dataset, '_result_5.mat');
load(filename);

% subOptNR = SVRGNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum);
% subOptA = SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorA, dataset, gridNum);
subOptIAG = IAG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorIAG, dataset, gridNum);
%
% subOptK = KatyushaNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum);
%
% subOptGD = GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorGD, batchSize, dataset, gridNum);
% subOpt = SVRG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
%
% subOptRR = SVRGRR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);

save(filename, 'subOpt', 'subOptNR', 'subOptK', 'subOptRR', 'subOptA', 'subOptGD', 'subOptIAG');

% Katyusha(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
