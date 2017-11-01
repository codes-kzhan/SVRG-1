function subOptimality = svm_SVRG(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit)

fprintf('Fitting data with SVRG ...\n');

% initialization
[d, n] = size(X);
iterNum = 2 * n;
lambda = objFunc.lambda;

eta = factor / objFunc.L
% eta = 5e-1

w = zeros(d, 1);

initCost = objFunc.PrintCost(w, ZT, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w - objFunc.optSolution).^2);

tstart = tic;

for s = 1:passes % for each epoch

    DIG_svm(w, Z, lambda, s, iterNum, factor);
    % print and plot
    cost = objFunc.PrintCost(w, ZT, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
        distance = sum((w - objFunc.optSolution).^2) / initDistance;
        subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
    end
    now = toc(tstart);
    if now > ourlimit
        break;
    end
end % epoch

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
% fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'DIG';
curve_style = ':';
% PlotTime(subOptimality, curve_style, label, dataset, gridNum);
% PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
