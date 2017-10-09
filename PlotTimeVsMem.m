res = [];
resNR = [];
resIAG = [];
nums = (2:2:16);
for i = 2:2:16
  filename = strcat('../data/HIGGS_MEM/', 'HIGGS_', num2str(i), 'G.mat');
  load(filename);
  res = [res; [i, subOpt(6, :)]];
  resNR = [resNR; [i, subOptNR(7, :)]];
end

plot(res(:, 1), res(:, 3), 'DisplayName', 'SVRG');
hold on
plot(resNR(:, 1), resNR(:, 3), 'DisplayName', 'DVRG');
legend('show')


for i = 2:2:16
  filename = strcat('../data/', 'HIGGS_IAG_', num2str(i), 'G.mat');
  load(filename);
  resIAG = [resIAG; [i, subOptIAG(7, :)]];
end

hold on
plot(resIAG(:, 1), resIAG(:, 3), 'DisplayName', 'IAG');
legend('show')
