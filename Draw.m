load('../data/criteo_32G.mat')
PlotTime(subOptNR, 'm-', 'SVRGNR-32G', 'criteo', 1);
PlotTime(subOpt, 'm-.', 'SVRG-32G', 'criteo', 1);

load('../data/criteo_16G.mat')
PlotTime(subOptNR, 'r-', 'SVRGNR-16G', 'criteo', 1);
PlotTime(subOpt, 'r-.', 'SVRG-16G', 'criteo', 1);

load('../data/criteo_8G.mat')
PlotTime(subOptNR, 'b-', 'SVRGNR-8G', 'criteo', 1);
PlotTime(subOpt, 'b-.', 'SVRG-8G', 'criteo', 1);
