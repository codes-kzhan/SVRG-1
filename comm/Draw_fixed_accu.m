sig = [];
sigm = [];
svrg = [];
kat = [];

for mem = 2:2:8
    load(strcat('../data/avazu_C_result_4_accu_', num2str(mem), 'G.mat'));
    svrg = [svrg; [mem, subOpt(end, 2)]];
    kat = [kat; [mem, subOptKatyusha(end, 2)]];
    sig = [sig; [mem, subOptNR(end, 2)]];
    sigm = [sigm; [mem, subOptK(end, 2)]];
end

plot(sig(:, 1), sig(:,2), 'o-', 'linewidth', 2, 'DisplayName', 'SIG');
hold on
plot(sigm(:, 1), sigm(:,2), 's-', 'linewidth', 2, 'DisplayName', 'SIG-M');
plot(svrg(:, 1), svrg(:,2), 'd-', 'linewidth', 2, 'DisplayName', 'SVRG');
plot(kat(:, 1), kat(:,2), '*-', 'linewidth', 2, 'DisplayName', 'Katyusha');

xlabel('phisical memory (GB)')
ylabel('time (s)')
legend('show')
