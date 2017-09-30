% criteo dataset
load('../data/criteo_32G.mat')
PlotTime(subOptNR, 'm-', 'SVRGNR-32G', 'criteo', 1);
PlotTime(subOpt, 'm-.', 'SVRG-32G', 'criteo', 1);

load('../data/criteo_16G.mat')
PlotTime(subOptNR, 'r-', 'SVRGNR-16G', 'criteo', 1);
PlotTime(subOpt, 'r-.', 'SVRG-16G', 'criteo', 1);

load('../data/criteo_8G.mat')
PlotTime(subOptNR, 'b-', 'SVRGNR-8G', 'criteo', 1);
PlotTime(subOpt, 'b-.', 'SVRG-8G', 'criteo', 1);

% avazu dataset
load('../data/avazu_8G.mat')
PlotTime(subOptNR, 'm-', 'SVRGNR-8G', 'avazu', 2);
PlotTime(subOpt, 'm-.', 'SVRG-8G', 'avazu', 2);

load('../data/avazu_4G.mat')
PlotTime(subOptNR, 'r-', 'SVRGNR-4G', 'avazu', 2);
PlotTime(subOpt, 'r-.', 'SVRG-4G', 'avazu', 2);

load('../data/avazu_2G.mat')
PlotTime(subOptNR, 'b-', 'SVRGNR-2G', 'avazu', 2);
PlotTime(subOpt, 'b-.', 'SVRG-2G', 'avazu', 2);

% HIGGS dataset
load('../data/HIGGS_8G.mat')
PlotTime(subOptNR, 'm-', 'SVRGNR-8G', 'HIGGS', 3);
PlotTime(subOpt, 'm-.', 'SVRG-8G', 'HIGGS', 3);

load('../data/HIGGS_4G.mat')
PlotTime(subOptNR, 'm-', 'SVRGNR-4G', 'HIGGS', 3);
PlotTime(subOpt, 'm-.', 'SVRG-4G', 'HIGGS', 3);

load('../data/HIGGS_2G.mat')
PlotTime(subOptNR, 'm-', 'SVRGNR-2G', 'HIGGS', 3);
PlotTime(subOpt, 'm-.', 'SVRG-2G', 'HIGGS', 3);
