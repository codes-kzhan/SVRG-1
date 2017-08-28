dataset = 'toy';
passes = 50;
factor = 0.1;   % step size = factor/L, where L denotes smoothness constant

%% preliminaries
[Xtrain, Xtest, ytrain, ytest] = LoadDataset(dataset);  % load dataset

lambda = 1e-4;
L = max(sum(Xtrain.^2, 2)) / 4 + lambda;
mu = lambda;
logCost = ObjFunc(lambda, L, mu);

% % find or load optimal solution
% objFuncType = '_logistic'
% filename = strcat('../data/', dataset, objFuncType, '_opt.mat');
% if exist(filename, 'file') ~= 2
%     wOpt = SVRG(logCost, Xtrain, ytrain, passes*10, factor);
%     save(filename, 'wOpt');
% else
%     load(filename, 'wOpt');
% end

%% SVRG
tstartSVRG = tic;
wOpt = SVRG(logCost, Xtrain, ytrain, passes, factor);
fprintf('training accuracy: %f\n', logCost.Score(wOpt, Xtrain, ytrain));
fprintf('test accuracy: %f\n', logCost.Score(wOpt, Xtest, ytest));
telapsedSVRG = toc(tstartSVRG);
