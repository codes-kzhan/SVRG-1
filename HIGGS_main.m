function HIGGS_main(mem_amount)
dataset = 'HIGGS';
gridNum = 3;
% dataset : toy, covtype, rcv1, avazu, MNIST.
lambda = 1e-8;
passes = 100000000;
factor = 0.1;
factorNR = 0.1;
alpha = 0.15;
alphaNR = 0.15;
factorIAG = 1e-6;
factorDIG = 0.1;
factorA = 0.1;
facotrGD = 1;
batchSize = 1;
ourlimit = 5000;

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

filename = strcat('../data/', dataset, '_C_result_6_', mem_amount, '.mat');
% load(filename);

subOptNR = svm_SVRGNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum, ourlimit);
subOptK = svm_KatyushaNR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alphaNR, batchSize, dataset, gridNum, ourlimit);
subOptIAG = svm_IAG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorIAG, dataset, gridNum, ourlimit);
subOpt = svm_SVRG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit);
subOptDIG = svm_DIG(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorDIG, dataset, gridNum, ourlimit, ourlimit);
subOptKatyusha = svm_Katyusha(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum, ourlimit);
subOptRR = svm_SVRGRR(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit);
subOptA = svm_SAGA(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factorA, dataset, gridNum, ourlimit);
subOptGD = svm_GD(svmCost, Xtrain, ytrain, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit);


save(filename, 'subOpt', 'subOptNR', 'subOptK', 'subOptDIG', 'subOptKatyusha', 'subOptIAG', 'subOptA', 'subOptGD', 'subOptRR');

% Katyusha(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(svmCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
