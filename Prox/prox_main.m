function prox_main(dataset, gridNum)
% dataset = 'MNIST';
% gridNum = 1;
% dataset : toy, covtype, rcv1, avazu, MNIST.


% dataset = 'covtype';
if strcmp(dataset, 'covtype')
    passes = 20;
    factorNR = 0.1;
    factor = 0.1;
    lambda2 = 1e-5;
    lambda1 = 1e-4;
    batchSize = 1;
elseif strcmp(dataset, 'rcv1')
    passes = 20;
    factorNR = 1/2;
    factor = 1/2;
    lambda2 = 1e-5;
    lambda1 = 1e-4;
    batchSize = 1;
elseif strcmp(dataset, 'MNIST')
    passes = 20;
    factorNR = 0.1;
    factor = 0.1;
    lambda2 = 1e-4;
    lambda1 = 1e-4;
    batchSize = 1;
end

%% preliminaries
filename = strcat('../data/', dataset, '_dataset.mat');
load(filename);
% [Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset
[n, d] = size(Xtrain);

L = max(sum(Xtrain.^2, 1)) / 4 + lambda2;
mu = lambda2;
logCost = LogL1(lambda2, lambda1, L, mu);

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_logistic_l2_l1_norm';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = prox_FindOptSolution(logCost, Xtrain, ytrain, Xtest, ytest, passes * 10, factor, batchSize);
    save(filename, 'wOpt');
else
    load(filename, 'wOpt');
end
logCost.optSolution = wOpt;
logCost.optCost = logCost.Cost(wOpt, Xtrain, ytrain)

%% have fun

% SVRG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SVRGNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SVRGNRM(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);

% factor = 1/2;
SVRGNRP(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum);
% factor = 1;
SVRGP(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);

% SVRGM(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
% box on
% grid on
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
