function wOpt = KatyushaNR(objFunc, X, y, Xtest, ytest, passes, factor, dataset, gridNum)

fprintf('Fitting data with KatyushaNR...\n');

% initialization
[d ,n] = size(X);
iterNum = n;
lambda = objFunc.lambda;

eta = factor / objFunc.L
% eta = 5e-1

wtilde = zeros(d, 1);


initCost = objFunc.PrintCost(wtilde, X, y, 0);
subOptimality = [0, 0];
tstart = tic;

tau2 = 1/2;
tau1 = min(sqrt(iterNum * objFunc.mu / 3 / objFunc.L), 1/2);
alpha = 1/(3 * tau1 * objFunc.L);
u = wtilde;
z = wtilde;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);

    for i = 1:iterNum
        idx = mod(i-1, n) + 1;
        % idx = randperm(n, 1);
        w = tau1 * z + tau2 * wtilde + (1 - tau2 - tau1) * u;

        Xtmp = X(:, idx);
        ytmp = y(idx);
        % new gradient
        tmpExp = exp(-ytmp .* (w'*Xtmp)')'; % 1-by-n vector
        % old gradient
        tmpExpTilde = exp(-ytmp .* (wtilde'*Xtmp)')'; % 1-by-n vector
        wDelta1 = mean(-ytmp' .* (1./(1 + tmpExpTilde) - 1./(1 + tmpExp)) .* Xtmp, 2);

        wDelta2 = wDelta1 + lambda * w;
        wDelta = wDelta2 + ntilde;

        znew = z - alpha * wDelta;
        u = w + tau1 * (znew - z);
        z = znew;
    end
    wtilde = u;

    % print and plot
    cost = objFunc.PrintCost(wtilde, X, y, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        logError = log10((cost - objFunc.optCost)/(initCost - objFunc.optCost));
        subOptimality = [subOptimality; [s, logError]];
    end
end % epoch

wOpt = wtilde;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'DVRG-K';
curve_style = '-.';
PlotCurve(subOptimality(:, 1), subOptimality(:, 2), curve_style, label, dataset, gridNum);

end  % function
