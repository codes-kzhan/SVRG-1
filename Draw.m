% MNIST dataset
load('../data/MNIST_result.mat')
PlotCurve(subOptNR, 'r-', 'DVRG', 'MNIST', 1);
PlotCurve(subOpt, 'm:', 'SVRG', 'MNIST', 1);
PlotCurve(subOptK, '-', 'DVRG-K', 'MNIST', 1);

% rcv1 dataset
load('../data/rcv1_result.mat')
PlotCurve(subOptNR, 'r-', 'DVRG', 'rcv1', 2);
PlotCurve(subOpt, 'm:', 'SVRG', 'rcv1', 2);
PlotCurve(subOptK, '-', 'DVRG-K', 'rcv1', 2);

% covtype dataset
load('../data/covtype_result.mat')
PlotCurve(subOptNR, 'r-', 'DVRG', 'covtype', 3);
PlotCurve(subOpt, 'm:', 'SVRG', 'covtype', 3);
PlotCurve(subOptK, '-', 'DVRG-K', 'covtype', 3);

% % criteo dataset
% load('../data/criteo_32G.mat')
% PlotTime(subOptNR, 'm-', 'SVRGNR-32G', 'criteo', 1);
% PlotTime(subOpt, 'm-.', 'SVRG-32G', 'criteo', 1);
%
% load('../data/criteo_16G.mat')
% PlotTime(subOptNR, 'r-', 'SVRGNR-16G', 'criteo', 1);
% PlotTime(subOpt, 'r-.', 'SVRG-16G', 'criteo', 1);
%
% load('../data/criteo_8G.mat')
% PlotTime(subOptNR, 'b-', 'SVRGNR-8G', 'criteo', 1);
% PlotTime(subOpt, 'b-.', 'SVRG-8G', 'criteo', 1);
%
% % avazu dataset
% load('../data/avazu_8G.mat')
% PlotTime(subOptNR, 'm-', 'SVRGNR-8G', 'avazu', 2);
% PlotTime(subOpt, 'm-.', 'SVRG-8G', 'avazu', 2);
%
% load('../data/avazu_4G.mat')
% PlotTime(subOptNR, 'r-', 'SVRGNR-4G', 'avazu', 2);
% PlotTime(subOpt, 'r-.', 'SVRG-4G', 'avazu', 2);
%
% load('../data/avazu_2G.mat')
% PlotTime(subOptNR, 'b-', 'SVRGNR-2G', 'avazu', 2);
% PlotTime(subOpt, 'b-.', 'SVRG-2G', 'avazu', 2);
%
% % HIGGS dataset
% load('../data/HIGGS_8G.mat')
% PlotTime(subOptNR, 'm-', 'SVRGNR-8G', 'HIGGS', 3);
% PlotTime(subOpt, 'm-.', 'SVRG-8G', 'HIGGS', 3);
%
% load('../data/HIGGS_4G.mat')
% PlotTime(subOptNR, 'r-', 'SVRGNR-4G', 'HIGGS', 3);
% PlotTime(subOpt, 'r-.', 'SVRG-4G', 'HIGGS', 3);
%
% load('../data/HIGGS_2G.mat')
% PlotTime(subOptNR, 'b-', 'SVRGNR-2G', 'HIGGS', 3);
% PlotTime(subOpt, 'b-.', 'SVRG-2G', 'HIGGS', 3);
