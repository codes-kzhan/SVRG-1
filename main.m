% function main(dataset, gridNum)
dataset = 'MNIST';
gridNum = 3;
% dataset : toy, covtype, rcv1, avazu, MNIST. HIGGS

if strcmp(dataset, 'covtype')
    passes = 20;
    factor = 1/2;
    factorNR = 1/2;
    alpha = 1;
    factorIAG = 1e-5;
    lambda = 1e-5;
    batchSize = 1;

% elseif strcmp(dataset, 'rcv1')
%     passes = 20;
%     factor = 3/8;
%     factorA = 3/8;
%     factorNR = 3/8;
%     alpha = 1;
%     lambda = 1e-5;
%     batchSize = 1;

% elseif strcmp(dataset, 'rcv1')
%     passes = 20;
%     factor = 3/8;
%     factorA = 1/2;
%     factorNR = 0.6875;
%     alpha = 1.5;
%     lambda = 1e-5;
%     batchSize = 1;

% elseif strcmp(dataset, 'rcv1')
%     passes = 20;
%     factor = 1;
%     factorA = 1;
%     factorIAG = 0.05;
%     factorNR = 3/8;
%     alpha = 1;
%     lambda = 1e-8;
%     batchSize = 1;

elseif strcmp(dataset, 'rcv1')
    passes = 20;
    factor = 1;
    factorA = 0.5;
    factorIAG = 0.1;
    factorNR = 1;
    alpha = 1;
    lambda = 1e-8;
    batchSize = 1;
elseif strcmp(dataset, 'MNIST')
    passes = 20;
    factor = 0.2;
    factorNR = 0.2;
    factorA = 0.1;
    factorIAG = 0.01;
    factorGD = 10;
    alphaNR = 1;
    alpha = 1;
    factorDIG = 1;
    lambda = 1e-4;
    batchSize = 1;
    ourlimit = 5000;

% elseif strcmp(dataset, 'avazu')
%     passes = 20;
%     factor = 1/4;
%     factorIAG = 0.05;
%     alpha = 1;
%     lambda = 1e-5;
%     batchSize = 64;

% elseif strcmp(dataset, 'avazu')
%     passes = 10;
%     factor = 10;
%     factorNR = 10;
%     factorIAG = 0.05;
%     factorDIG = 1;
%     alpha = 9;
%     alphaNR = 9;
%     lambda = 1e-8;
%     batchSize = 64;
%     ourlimit = 5000;

elseif strcmp(dataset, 'avazu')
    passes = 5;
    factor = 1;
    factorNR = 10;
    factorIAG = 5e-6;
    factorDIG = 1;
    alpha = 9;
    alphaNR = 9;
    lambda = 1e-8;
    batchSize = 1;
    ourlimit = 5000;

elseif strcmp(dataset, 'criteo')
    passes = 1;
    factor = 1/2;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 1;
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

% filename = strcat('../data/', dataset, '_dataset.mat');
% load(filename);

L = max(sum(Xtrain.^2, 1)) / 4 + lambda;
mu = lambda;
logCost = LR(lambda, L, mu);

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

% filename = strcat('../data/', dataset, '_result_4_1e-8_2G.mat');
% load(filename);

subOptKatyusha = Katyusha(logCost, Xtrain, ytrain, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum, ourlimit);
% subOptDIG = DIG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorDIG, batchSize, dataset, gridNum, ourlimit);
% subOptNR = SVRGNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum);
% subOptK = KatyushaNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, alphaNR, batchSize, dataset, gridNum);
subOpt = SVRG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% subOptA = SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorA, dataset, gridNum);
% subOptIAG = IAG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorIAG, dataset, gridNum);
% subOptGD = GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorGD, batchSize, dataset, gridNum);
% subOptRR = SVRGRR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);

% save(filename, 'subOpt', 'subOptNR', 'subOptK', 'subOptDIG', 'subOptKatyusha');

%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
