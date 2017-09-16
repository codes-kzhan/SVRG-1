function PlotCurve(data_point, curve_style, label, name, gridNum)

subplot(2, 3, gridNum)
% subplot(1, 1, gridNum)
title(name)
semilogy(data_point(:, 1), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('epoch')
ylabel('suboptimality')
legend('show')
% box on
% grid on
hold on


subplot(2, 3, gridNum + 3)
semilogy(data_point(:, 1), data_point(:, 4), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('epoch')
ylabel('distance to opt')
legend('show')
% box on
% grid on
hold on
