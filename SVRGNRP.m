function wOpt = SVRGP(objFunc, X, y, Xtest, ytest, passes, factor)

tstart = tic;
fprintf('Fitting data with SVRGNR-Prox ...\n');

% initialization
[n ,d] = size(X);
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
        wDelta = objFunc.Gradient(w, X(idx, :), y(idx)) - objFunc.Gradient(wtilde, X(idx, :), y(idx)) + objFunc.lambda2 * w + ntilde;
        w = prox(w - eta*wDelta, eta, 1, objFunc.lambda1);
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


label = 'SVRGNR-Prox';
curve_style = 'r-.';
PlotCurve(0:validPoints-1, subOptimality(1:validPoints), curve_style, label);

end  % function
