%fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
% fig = figure();
% main('MNIST', 1);
main('rcv1', 2);
svm_main('covtype', 3);


% figname = './whole.png';
% saveas(fig, figname);
% close(fig);
