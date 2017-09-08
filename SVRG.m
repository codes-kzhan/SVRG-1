function wOpt = SVRG(objFunc, X, y, Xtest, ytest, passes, factor)

tstart = tic;
fprintf('Fitting data with SVRG ...\n');

% initialization
[d, n] = size(X);
batchSize = 64;
iterNum = round(n/batchSize);
lambda = objFunc.lambda;

eta = factor / objFunc.L
% eta = 5e-1

if issparse(X)
    wtilde = sparse(d, 1);
else
    wtilde = zeros(d, 1);
end
w = wtilde;

initCost = objFunc.PrintCost(wtilde, X, y, 0);
subOptimality = [0, 0];

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);

    for i = 1:iterNum
        idx = randperm(n, batchSize);
        Xtmp = X(:, idx);
        ytmp = y(idx);
        % new gradient
        tmpExp = exp(-ytmp .* (w'*Xtmp)')'; % 1-by-n vector
        % old gradient
        tmpExpTilde = exp(-ytmp .* (wtilde'*Xtmp)')'; % 1-by-n vector
        wDelta1 = mean(-ytmp' .* (tmpExp./(1 + tmpExp) - tmpExpTilde./(1 + tmpExpTilde)) .* Xtmp, 2);

        wDelta2 = wDelta1 + lambda * w;
        wDelta3 = wDelta2 + ntilde;
        w = w - eta * wDelta3;
    end
    wtilde = w;

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


label = 'SVRG';
curve_style = 'm-';
PlotCurve(subOptimality(:, 1), subOptimality(:, 2), curve_style, label);

end  % function
