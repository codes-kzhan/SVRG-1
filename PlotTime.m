function PlotTime(data_point, curve_style, label, name, gridNum)

% subplot(2, 2, gridNum)
subplot(1, 2, 1)
title(name)
hold on;
plot(data_point(:, 1), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('epoch')
ylabel('log-suboptimality')
legend('show')

subplot(1, 2, 2)
title(name)
hold on;
plot(data_point(:, 2), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('time(seconds)')
ylabel('log-suboptimality')
legend('show')
