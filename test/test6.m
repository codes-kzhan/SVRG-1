% function main(dataset, gridNum)
dataset = 'HIGGS';
gridNum = 3;
% dataset : toy, covtype, rcv1, avazu, MNIST.

if strcmp(dataset, 'covtype')
    passes = 20;
    factorNR = 0.25;
    factor = 0.25;
    factorA = 0.14;
    factorIAG = 1e-5;
    alpha = 0.75;
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
    factorNR = 0.1;
    factor = 0.1;
    alpha = 0.2;
    lambda = 1e-4;
    batchSize = 1;
elseif strcmp(dataset, 'avazu')
    passes = 20;
    factorNR = 1;
    factor = 1;
    alpha = 1;
    lambda = 1e-5;
    batchSize = 64;
elseif strcmp(dataset, 'criteo')
    passes = 20;
    factor = 1/2;
    lambda = 1e-5;
    batchSize = 64;
elseif strcmp(dataset, 'HIGGS')
    passes = 6;
    factor = 0.05;
    factorIAG = 1e-6;
    factorNR = 0.05;
    lambda = 1e-5;
    batchSize = 1;
elseif strcmp(dataset, 'ijcnn1')
    passes = 20;
    factor = 0.1;
    factorNR = 0.05;
    alpha = 0.09;
    lambda = 1e-5;
    batchSize = 1;
end
%% preliminaries
% [Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset
filename = strcat('../data/', dataset, '_dataset.mat');
load(filename);
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

% filename = strcat('../data/', dataset, '_result_5.mat');
% load(filename);

% svm_KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum);
% subOptRR = svm_SVRGRR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);

% subOptK = svm_KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum);

% subOptK = svm_Katyusha(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum);

% subOptA = svm_SAGA(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorA, dataset, gridNum);
subOptIAG = svm_IAG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorIAG, dataset, gridNum);


% subOpt = svm_SVRG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% subOptGD = svm_GD(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% subOptNR = svm_SVRGNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum);

filename = strcat('../data/', dataset, '_IAG_6G.mat');
save(filename, 'subOptIAG');

% Katyusha(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
