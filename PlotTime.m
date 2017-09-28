function PlotTime(data_point, curve_style, label, name, gridNum)

% subplot(2, 2, gridNum)
subplot(1, 2, 1)
title(name)
semilogy(data_point(:, 1), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
hold on;
xlabel('epoch')
ylabel('log-suboptimality')
legend('show')

subplot(1, 2, 2)
title(name)
semilogy(data_point(:, 2), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
hold on;
xlabel('time(seconds)')
ylabel('log-suboptimality')
legend('show')
