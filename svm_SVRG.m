function subOptimality = svm_SVRG(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum)

fprintf('Fitting data with SVRG ...\n');

% initialization
[d, n] = size(X);
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

initCost = objFunc.PrintCost(wtilde, ZT, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((wtilde - objFunc.optSolution).^2);

tstart = tic;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, Z, ZT);
    for i = 1:iterNum
        idx = randperm(n, batchSize);
        Ztmp = Z(:, idx);
        ZTtmp = Ztmp';

        tmpDeltaG = Ztmp * (max(1 + ZTtmp * w, 0) - max(1 + ZTtmp * wtilde, 0)) * 2/batchSize;

        wDelta1 = tmpDeltaG + lambda * w;
        wDelta2 = wDelta1 + ntilde;
        w = w - eta * wDelta2;
    end
    wtilde = w;

    % print and plot
    cost = objFunc.PrintCost(wtilde, ZT, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
        distance = sum((wtilde - objFunc.optSolution).^2) / initDistance;
        subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
    end
end % epoch

wOpt = wtilde;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'SVRG';
curve_style = 'm:';
% PlotTime(subOptimality, curve_style, label, dataset, gridNum);
PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
