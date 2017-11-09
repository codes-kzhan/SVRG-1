label = 'criteo';
grid = 1;
mem = 64;
for i = 1:3
    mem = mem/2;
    fn = strcat('../data/criteo_C_result_2_time_', num2str(mem), 'G.mat');
    load(fn);
    PlotLarge(subOptNR, '-', strcat('SIG,', num2str(mem), 'G'), label, grid);
    PlotLarge(subOpt, ':', strcat('SVRG,', num2str(mem), 'G'), label, grid);
    xlim([0, 1000]);
end

load('../data/bar_time.mat');
plotBarStackGroups(time3, {'8GB', '16GB', '32GB'});
legend('show')
xlabel('phisical memory (GB)')
ylabel('time (s)')
