% dataset : toy, covtype, rcv1, avazu, MNIST.

% dataset = 'covtype';
% passes = 20;
% factor = 0.1;
% lambda2 = 1e-5;
% lambda1 = 1e-4;

% dataset = 'rcv1';
% passes = 20;
% factor = 1;
% lambda2 = 1e-5;
% lambda1 = 1e-4;

dataset = 'MNIST';
passes = 50;
factor = 1/4;
lambda2 = 1e-4;
lambda1 = 1e-4;

%% preliminaries
[Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset
[n, d] = size(Xtrain);

L = max(sum(Xtrain.^2, 2)) / 4;
mu = lambda1;
logCost = LogL1(lambda2, lambda1, L, mu);

fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% find or load optimal solution
objFuncType = '_logistic_l2_l1';
filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
if exist(filename, 'file') ~= 2
    wOpt = FindOptSolution(logCost, Xtrain, ytrain, Xtest, ytest, passes * 10, factor);
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

factor = 1/2;
SVRGNRP(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
factor = 1;
SVRGP(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);

% SVRGM(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SAGA(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% SGD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
% GD(logCost, Xtrain, ytrain, Xtest, ytest, passes, factor);
%% save figure and exit
box on
grid on
figname = strcat('./', dataset, objFuncType, '.png');
saveas(fig, figname);
close(fig);
