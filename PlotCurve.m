function PlotCurve(x, y, curve_style, label)

subplot(1, 3, 3)
hold on;
plot(x, y, curve_style, 'linewidth', 2, 'DisplayName', label);
legend('show')
