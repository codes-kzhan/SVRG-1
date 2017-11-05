function PlotTime(data_point, curve_style, label, name, gridNum)

% subplot(3, 3, gridNum)
% title(name)
% semilogy(data_point(:, 1), data_point(:, 4), curve_style, 'linewidth', 2, 'DisplayName', label);
% % semilogy(data_point(:, 1), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
% hold on;
% xlabel('epoch')
% ylabel('distance to opt')
% % ylabel('log-suboptimality')
% legend('show')

% subplot(3, 3, gridNum+3)
% title(name)
% % semilogy(data_point(:, 2), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
% semilogy(data_point(:, 2), data_point(:, 4), curve_style, 'linewidth', 2, 'DisplayName', label);
% hold on;
% xlabel('time(seconds)')
% % ylabel('log-suboptimality')
% ylabel('distance to opt')
% legend('show')

subplot(3, 3, gridNum+6)
% subplot(1, 1, gridNum)
title(name)
semilogy(data_point(:, 2), data_point(:, 3), curve_style, 'linewidth', 2, 'DisplayName', label);
xlabel('time')
ylabel('suboptimality')
legend('show')
% box on
% grid on
hold on
