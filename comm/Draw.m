% MNIST dataset
label = 'MNIST';
grid = 1;
load('../data/MNIST_C_result_6_32G.mat');
PlotCurve(subOptNR, '-', 'SIG', label, grid);
PlotCurve(subOpt, ':', 'SVRG', label, grid);
PlotCurve(subOptKatyusha, ':', 'SVRG-K', label, grid);
PlotCurve(subOptK, '-', 'SIG-K', label, grid);
PlotCurve(subOptRR, ':', 'SVRG-RR', label, grid);
PlotCurve(subOptA, ':', 'SAGA-RR', 'MNIST', grid);
PlotCurve(subOptIAG, ':', 'IAG', label, grid);
PlotCurve(subOptGD, '-.', 'GD', label, grid);

% rcv1 dataset
label = 'rcv1';
grid = 2;
load('../data/rcv1_C_result_6_32G.mat');
PlotCurve(subOptNR, '-', 'SIG', label, grid);
PlotCurve(subOpt, ':', 'SVRG', label, grid);
PlotCurve(subOptKatyusha, ':', 'SVRG-K', label, grid);
PlotCurve(subOptK, '-', 'SIG-K', label, grid);
PlotCurve(subOptRR, ':', 'SVRG-RR', label, grid);
PlotCurve(subOptA, ':', 'SAGA-RR', 'MNIST', grid);
PlotCurve(subOptIAG, ':', 'IAG', label, grid);
PlotCurve(subOptGD, '-.', 'GD', label, grid);

% % covtype dataset
% load('../data/covtype_result_5.mat')
% PlotCurve(subOptNR, '-', 'SIG', 'covtype', 3);
% PlotCurve(subOpt, ':', 'SVRG', 'covtype', 3);
% PlotCurve(subOptK, '-', 'SIG-K', 'covtype', 3);
% PlotCurve(subOptRR, ':', 'SVRG-RR', 'covtype', 3);
% PlotCurve(subOptA, ':', 'SAGA-RR', 'covtype', 3);
% PlotCurve(subOptIAG, ':', 'IAG', 'covtype', 3);
% PlotCurve(subOptGD, '-.', 'GD', 'covtype', 3);
% PlotCurve(subOptKatyusha, ':', 'Katyusha', 'covtype', 3);

% a9a dataset
label = 'a9a';
grid = 3;
load('../data/a9a_C_result_6_32G.mat');
PlotCurve(subOptNR, '-', 'SIG', label, grid);
PlotCurve(subOpt, ':', 'SVRG', label, grid);
PlotCurve(subOptKatyusha, ':', 'SVRG-K', label, grid);
PlotCurve(subOptK, '-', 'SIG-K', label, grid);
PlotCurve(subOptRR, ':', 'SVRG-RR', label, grid);
PlotCurve(subOptA, ':', 'SAGA-RR', 'MNIST', grid);
PlotCurve(subOptIAG, ':', 'IAG', label, grid);
PlotCurve(subOptGD, '-.', 'GD', label, grid);

% % criteo dataset
% load('../data/criteo_32G.mat')
% PlotTime(subOptNR, '-', 'SIG, 32G', 'criteo', 1);
% PlotTime(subOpt, ':', 'SVRG, 32G', 'criteo', 1);
%
% load('../data/criteo_16G.mat')
% PlotTime(subOptNR, '-', 'SIG, 16G', 'criteo', 1);
% PlotTime(subOpt, ':', 'SVRG, 16G', 'criteo', 1);
%
% load('../data/criteo_8G.mat')
% PlotTime(subOptNR, '-', 'SIG, 8G', 'criteo', 1);
% PlotTime(subOpt, ':', 'SVRG, 8G', 'criteo', 1);
%
% % avazu dataset
% load('../data/avazu_8G.mat')
% PlotTime(subOptNR, '-', 'SIG, 8G', 'avazu', 2);
% PlotTime(subOpt, ':', 'SVRG, 8G', 'avazu', 2);
%
% load('../data/avazu_4G.mat')
% PlotTime(subOptNR, '-', 'SIG, 4G', 'avazu', 2);
% PlotTime(subOpt, ':', 'SVRG, 4G', 'avazu', 2);
%
% load('../data/avazu_2G.mat')
% PlotTime(subOptNR, '-', 'SIG, 2G', 'avazu', 2);
% PlotTime(subOpt, ':', 'SVRG, 2G', 'avazu', 2);
%
% % HIGGS dataset
% load('../data/HIGGS_8G.mat')
% PlotTime(subOptNR, '-', 'SIG, 8G', 'HIGGS', 3);
% PlotTime(subOpt, ':', 'SVRG, 8G', 'HIGGS', 3);
%
% load('../data/HIGGS_4G.mat')
% PlotTime(subOptNR, '-', 'SIG, 4G', 'HIGGS', 3);
% PlotTime(subOpt, ':', 'SVRG, 4G', 'HIGGS', 3);
%
% load('../data/HIGGS_2G.mat')
% PlotTime(subOptNR, '-', 'SIG, 2G', 'HIGGS', 3);
% PlotTime(subOpt, ':', 'SVRG, 2G', 'HIGGS', 3);
