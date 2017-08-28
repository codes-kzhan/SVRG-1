% dataset : toy, covtype, rcv1, avazu, etc.
dataset = 'covtype';
passes = 50;
factor = 1/2;   % step size = factor/L, where L denotes smoothness constant
lambda = 1e-5;

%% preliminaries
[Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset

L = max(sum(Xtrain.^2, 2)) / 4 + lambda;
mu = lambda;
logCost = ObjFunc(lambda, L, mu);

% find or load optimal solution
objFuncType = '_logistic';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = FindOptSolution(logCost, Xtrain, ytrain, Xtest, ytest, passes * 50, factor);
    save(filename, 'wOpt');
else
    load(filename, 'wOpt');
end
logCost.optSolution = wOpt;
logCost.optCost = logCost.Cost(wOpt, Xtrain, ytrain);

%% have fun

SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SVRG(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);

figname = strcat('./', dataset, objFuncType, '.png');
saveas(fig, figname);
close(fig);
