function PlotCurve(x, y, curve_style, label, name, gridNum)

% subplot(2, 2, gridNum)
subplot(1, 1, gridNum)
title(name)
hold on;
plot(x, y, curve_style, 'linewidth', 2, 'DisplayName', label);
legend('show')
