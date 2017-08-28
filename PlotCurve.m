function PlotCurve(x, y, curve_style, label)

subplot()
hold on;
plot(x, y, curve_style, 'linewidth', 2, 'DisplayName', label);
legend('show')
