% function main(dataset, gridNum)
dataset = 'a9a';
gridNum = 3;
% dataset : toy, covtype, rcv1, avazu, MNIST.
passes = 100;
factor = 0.4;
factorNR = 0.35;
alpha = 0.5;
alphaNR = 0.6;
factorIAG = 1e-4;
factorA = 0.6;
factorDIG = 1;
facotrGD = 1;
lambda = 1e-7;
batchSize = 1;
ourlimit = 20;

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

% filename = strcat('../data/', dataset, '_result_5.mat');
% load(filename);

% subOptRR = svm_SVRGRR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit);
% subOptNR = svm_SVRGNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum, ourlimit);
% subOptK = svm_KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alphaNR, batchSize, dataset, gridNum, ourlimit);
% subOptIAG = svm_IAG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorIAG, dataset, gridNum, ourlimit);
subOpt = svm_SVRG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);
% subOptDIG = svm_DIG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorDIG, dataset, gridNum, ourlimit);
% subOptKatyusha = svm_Katyusha(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum, ourlimit);
% subOptA = svm_SAGA(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorA, dataset, gridNum, ourlimit);
% subOptGD = svm_GD(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit);


% save(filename, 'subOpt', 'subOptNR', 'subOptK', 'subOptRR', 'subOptA', 'subOptGD');

% Katyusha(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
