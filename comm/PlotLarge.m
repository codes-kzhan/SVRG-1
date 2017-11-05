function PlotLarge(data_point, curve_style, label, name, gridNum)

subplot(1, 3, gridNum)
% subplot(1, 1, gridNum)
title(name)
semilogy(data_point(:, 2), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('time (s)')
ylabel('suboptimality')
legend('show')
% box on
% grid on
hold on
