%fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
fig = figure();
main('covtype', 1);
main('rcv1', 2);
main('MNIST', 3);
main('avazu', 4);

figname = './whole.png';
saveas(fig, figname);
% close(fig);
