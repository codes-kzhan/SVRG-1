function PlotCurve(x, y, curve_style, label, name, gridNum)

subplot(2, 2, gridNum)
title(name)
hold on;
plot(x, y, curve_style, 'linewidth', 2, 'DisplayName', label);
legend('show')
