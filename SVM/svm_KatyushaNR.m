function subOptimality = svm_KatyushaNR(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum)

fprintf('Fitting data with KatyushaNR...\n');

% initialization
[d, n] = size(X);
iterNum = n;
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

tau2 = 1/2;
u = wtilde;
z = wtilde;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, Z, ZT);
    tau1 = 2/(s+4);
    alpha = factor/(3 * tau1 * objFunc.L);

    for i = 1:iterNum
        idx = mod(i-1, n) + 1;
        w = tau1 * z + tau2 * wtilde + (1 - tau2 - tau1) * u;

        Ztmp = Z(:, idx);
        ZTtmp = ZT(idx, :);

        tmpDeltaG = Ztmp * (max(1 + ZTtmp * w, 0) - max(1 + ZTtmp * wtilde, 0)) * 2/batchSize;

        wDelta1 = tmpDeltaG + lambda * w;
        wDelta2 = wDelta1 + ntilde;
        znew = z - alpha * wDelta2;
        u = w + tau1 * (znew - z);
        z = znew;
    end
    wtilde = u;

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


label = 'DVRK-K';
curve_style = '-';
% PlotTime(subOptimality, curve_style, label, dataset, gridNum);
PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
