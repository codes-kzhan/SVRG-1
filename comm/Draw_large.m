% avazu dataset
load('../data/avazu_result_4_1e-8_4G.mat');
PlotLarge(subOptNR, 'b-', 'SIG', 'avazu', 1);
PlotLarge(subOpt, 'b:', 'SVRG', 'avazu', 1);
PlotLarge(subOptK, '-', 'SIG-M', 'avazu', 1);
PlotLarge(subOptKatyusha, ':', 'Katyusha', 'avazu', 1);
PlotLarge(subOptDIG, ':', 'DIG', 'avazu', 1);
% PlotLarge(subOptRR, ':', 'SVRG-RR', 'avazu', 1);
% PlotLarge(subOptA, ':', 'SAGA-RR', 'avazu', 1);
PlotLarge(subOptIAG, ':', 'IAG', 'avazu', 1);
% PlotLarge(subOptGD, '-.', 'GD', 'avazu', 1);
xlim([0, 5000]);

% HIGGS dataset
load('../data/HIGGS_1e-8_2G_K.mat');
PlotLarge(subOptNR, '-', 'SIG', 'HIGGS', 2);
PlotLarge(subOpt, ':', 'SVRG', 'HIGGS', 2);
PlotLarge(subOptK, '-', 'SIG-M', 'HIGGS', 2);
PlotLarge(subOptKatyusha, ':', 'Katyusha', 'HIGGS', 2);
PlotLarge(subOptDIG, ':', 'DIG', 'HIGGS', 2);
% PlotLarge(subOptRR, ':', 'SVRG-RR', 'HIGGS', 2);
% PlotLarge(subOptA, ':', 'SAGA-RR', 'HIGGS', 2);
PlotLarge(subOptIAG, ':', 'IAG', 'HIGGS', 2);
% PlotLarge(subOptGD, '-.', 'GD', 'HIGGS', 2);
xlim([0, 1000]);

% criteo dataset
load('../data/criteo_C_result_6_8G.mat');
PlotLarge(subOptNR, 'b-', 'SIG', 'criteo', 1);
PlotLarge(subOpt, 'b:', 'SVRG', 'criteo', 1);
PlotLarge(subOptK, '-', 'SIG-K', 'criteo', 1);
PlotLarge(subOptKatyusha, ':', 'Katyusha', 'criteo', 1);
PlotLarge(subOptDIG, ':', 'DIG', 'criteo', 1);
% PlotLarge(subOptRR, ':', 'SVRG-RR', 'avazu', 1);
% PlotLarge(subOptA, ':', 'SAGA-RR', 'avazu', 1);
PlotLarge(subOptIAG, ':', 'IAG', 'criteo', 1);
% PlotLarge(subOptGD, '-.', 'GD', 'avazu', 1);
xlim([0, 5000]);

% % kdd2010 dataset
% load('../data/kddb_result.mat');
% PlotLarge(subOptNR, '-', 'SIG', 'kdd2010', 3);
% PlotLarge(subOpt, ':', 'SVRG', 'kdd2010', 3);
% PlotLarge(subOptK, '-', 'SIG-K', 'kdd2010', 3);
% xlim([0, 10000]);
