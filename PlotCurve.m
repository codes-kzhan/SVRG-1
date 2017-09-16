function PlotCurve(data_point, curve_style, label, name, gridNum)

subplot(2, 3, gridNum)
% subplot(1, 1, gridNum)
title(name)
hold on;
semilogy(data_point(:, 1), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('epoch')
ylabel('suboptimality')
legend('show')


subplot(2, 3, gridNum + 3)
hold on;
plot(data_point(:, 1), data_point(:, 4), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('epoch')
ylabel('distance to opt')
legend('show')
