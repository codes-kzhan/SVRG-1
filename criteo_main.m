function main(mem_amount)
dataset = 'criteo';
gridNum = 3;
% dataset : toy, covtype, rcv1, avazu, MNIST. HIGGS

passes = 5000;
factor = 1/2;
factorNR = 1/2;
% factorA = 0.1;
factorIAG = 1e-6;
% factorGD = 10;
alphaNR = 1;
alpha = 1;
factorDIG = 1e1;
lambda = 1e-9;
batchSize = 1;
ourlimit = 5000;

%% preliminaries
% [Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset

filename = strcat('../data/', dataset, '_dataset.mat');
load(filename);

L = sum(Xtrain(:, 1).^2, 1) / 4 + lambda;
mu = lambda;
logCost = LR(lambda, L, mu);

% fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_logistic_norm';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = FindOptSolution(logCost, Xtrain, ytrain, Xtest, ytest, passes*10, factor, batchSize);
    save(filename, 'wOpt');
else
    load(filename, 'wOpt');
end
logCost.optSolution = wOpt;
logCost.optCost = logCost.Cost(wOpt, Xtrain, ytrain)

%% have fun

filename = strcat('../data/', dataset, '_C_result_6_', mem_amount, '.mat');
% load(filename);

subOpt = SVRG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit);
subOptKatyusha = Katyusha(logCost, Xtrain, ytrain, Xtest, ytest, passes, alpha, batchSize, dataset, gridNum, ourlimit);
subOptDIG = DIG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorDIG, batchSize, dataset, gridNum, ourlimit);
subOptNR = SVRGNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorNR, batchSize, dataset, gridNum, ourlimit);
subOptK = KatyushaNR(logCost, Xtrain, ytrain, Xtest, ytest, passes, alphaNR, batchSize, dataset, gridNum, ourlimit);
% subOptA = SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorA, dataset, gridNum);
subOptIAG = IAG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorIAG, dataset, gridNum, ourlimit);
% subOptGD = GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factorGD, batchSize, dataset, gridNum);
% subOptRR = SVRGRR(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor, batchSize, dataset, gridNum);

save('tmp_criteo_main.mat', 'subOpt', 'subOptNR', 'subOptK', 'subOptDIG', 'subOptKatyusha', 'subOptIAG');
save(filename, 'subOpt', 'subOptNR', 'subOptK', 'subOptDIG', 'subOptKatyusha', 'subOptIAG');

%% save figure and exit
% figname = strcat('./', dataset, objFuncType, '.png');
% saveas(fig, figname);
% close(fig);
