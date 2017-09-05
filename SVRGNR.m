function wOpt = SVRGNR(objFunc, X, y, Xtest, ytest, passes, factor)

tstart = tic;
fprintf('Fitting data with SVRG-NR ...\n');

% initialization
[d ,n] = size(X);
iterNum = n;
subOptimality = zeros(passes, 1);
validPoints = 0;

eta = factor / objFunc.L
% eta = 5e-1

wtilde = zeros(d, 1);
w = wtilde;

initCost = objFunc.PrintCost(wtilde, X, y, 0);
validPoints = validPoints + 1;
subOptimality(1) = 0;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);

    for i = 1:iterNum
        idx = i;

        Xtmp = X(:, idx);
        ytmp = y(idx);
        % new gradient
        tmpExp = exp(-ytmp .* (w'*Xtmp)')'; % 1-by-n vector
        newGradient = (-ytmp' .* tmpExp .* Xtmp ) ./ (1 + tmpExp);
        % old gradient
        tmpExp = exp(-ytmp .* (wtilde'*Xtmp)')'; % 1-by-n vector
        oldGradient = (-ytmp' .* tmpExp .* Xtmp ) ./ (1 + tmpExp);

        wDelta = newGradient - oldGradient + objFunc.lambda * w + ntilde;
        w = w - eta * wDelta;
    end
    wtilde = w;

    % print and plot
    cost = objFunc.PrintCost(wtilde, X, y, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        validPoints = validPoints + 1;
        subOptimality(validPoints) = log10((cost - objFunc.optCost)/(initCost - objFunc.optCost));
    end
end % epoch

wOpt = wtilde;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'SVRG-NR';
curve_style = 'r-';
PlotCurve(0:validPoints-1, subOptimality(1:validPoints), curve_style, label);

end  % function
