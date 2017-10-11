% MNIST dataset
load('../data/MNIST_result_5.mat')
PlotCurve(subOptNR, '-', 'SIG', 'MNIST', 1);
PlotCurve(subOpt, ':', 'SVRG', 'MNIST', 1);
PlotCurve(subOptKatyusha, ':', 'Katyusha', 'MNIST', 1);
PlotCurve(subOptK, '-', 'SIG-M', 'MNIST', 1);
PlotCurve(subOptRR, ':', 'SVRG-RR', 'MNIST', 1);
PlotCurve(subOptA, ':', 'SAGA-RR', 'MNIST', 1);
PlotCurve(subOptIAG, ':', 'IAG', 'MNIST', 1);
PlotCurve(subOptDIG, ':', 'DIG', 'MNIST', 1);
PlotCurve(subOptGD, '-.', 'GD', 'MNIST', 1);

% rcv1 dataset
load('../data/rcv1_result_5_1e-8.mat')
PlotCurve(subOptNR, '-', 'SIG', 'rcv1', 2);
PlotCurve(subOpt, ':', 'SVRG', 'rcv1', 2);
PlotCurve(subOptKatyusha, ':', 'Katyusha', 'rcv1', 2);
PlotCurve(subOptK, '-', 'SIG-M', 'rcv1', 2);
PlotCurve(subOptRR, ':', 'SVRG-RR', 'rcv1', 2);
PlotCurve(subOptA, ':', 'SAGA-RR', 'rcv1', 2);
PlotCurve(subOptIAG, ':', 'IAG', 'rcv1', 2);
PlotCurve(subOptDIG, ':', 'DIG', 'rcv1', 2);
PlotCurve(subOptGD, '-.', 'GD', 'rcv1', 2);

% ijcnn1 dataset
load('../data/ijcnn1_result_5.mat')
PlotCurve(subOptNR, '-', 'SIG', 'ijcnn1', 3);
PlotCurve(subOpt, ':', 'SVRG', 'ijcnn1', 3);
PlotCurve(subOptKatyusha, ':', 'Katyusha', 'ijcnn1', 3);
PlotCurve(subOptK, '-', 'SIG-M', 'ijcnn1', 3);
PlotCurve(subOptRR, ':', 'SVRG-RR', 'ijcnn1', 3);
PlotCurve(subOptA, ':', 'SAGA-RR', 'ijcnn1', 3);
PlotCurve(subOptIAG, ':', 'IAG', 'ijcnn1', 3);
PlotCurve(subOptDIG, ':', 'DIG', 'ijcnn1', 3);
PlotCurve(subOptGD, '-.', 'GD', 'ijcnn1', 3);

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
