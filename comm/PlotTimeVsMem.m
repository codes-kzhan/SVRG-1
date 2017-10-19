res = [];
resNR = [];
nums = (2:2:16);
for i = 2:2:16
  filename = strcat('../data/', 'HIGGS_', num2str(i), 'G.mat');
  load(filename);
  res = [res; [i, subOpt(6, :)]];
  resNR = [resNR; [i, subOptNR(7, :)]];
end

plot(res(:, 1), res(:, 3), 'DisplayName', 'SVRG');
hold on
plot(resNR(:, 1), resNR(:, 3), 'DisplayName', 'DVRG');
legend('show')
