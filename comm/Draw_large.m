% avazu dataset
label = 'avazu';
grid = 1;
load('../data/avazu_C_result_6_2G.mat');
PlotLarge(subOptNR, '-', 'SIG', label, grid);
PlotLarge(subOpt, ':', 'SVRG', label, grid);
PlotLarge(subOptKatyusha, ':', 'SVRG-K', label, grid);
PlotLarge(subOptK, '-', 'SIG-K', label, grid);
PlotLarge(subOptIAG, ':', 'IAG', label, grid);
PlotLarge(subOptDIG, ':', 'DIG', label, grid);
xlim([0, 5000]);

% HIGGS dataset
label = 'HIGGS';
grid = 3;
load('../data/HIGGS_C_result_6_2G.mat');
PlotLarge(subOptNR, '-', 'SIG', label, grid);
PlotLarge(subOpt, ':', 'SVRG', label, grid);
PlotLarge(subOptKatyusha, ':', 'SVRG-K', label, grid);
PlotLarge(subOptK, '-', 'SIG-K', label, grid);
PlotLarge(subOptIAG, ':', 'IAG', label, grid);
PlotLarge(subOptDIG, ':', 'DIG', label, grid);
xlim([0, 3500]);

% criteo dataset
label = 'criteo';
grid = 2;
load('../data/criteo_C_result_6_8G.mat');
PlotLarge(subOptNR, '-', 'SIG', label, grid);
PlotLarge(subOpt, ':', 'SVRG', label, grid);
PlotLarge(subOptKatyusha, ':', 'SVRG-K', label, grid);
PlotLarge(subOptK, '-', 'SIG-K', label, grid);
PlotLarge(subOptIAG, ':', 'IAG', label, grid);
PlotLarge(subOptDIG, ':', 'DIG', label, grid);
xlim([0, 5000]);

% % kdd2010 dataset
% load('../data/kddb_result.mat');
% PlotLarge(subOptNR, '-', 'SIG', 'kdd2010', 3);
% PlotLarge(subOpt, ':', 'SVRG', 'kdd2010', 3);
% PlotLarge(subOptK, '-', 'SIG-K', 'kdd2010', 3);
% xlim([0, 10000]);
