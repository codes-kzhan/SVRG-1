function subOptimality = GD(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit)

fprintf('Fitting data with GD ...\n');

% initialization
[d, n] = size(X);
iterNum = passes*2;
lambda = objFunc.lambda;

eta = 1 / objFunc.L

w = zeros(d, 1);
% wOpt = 0;

initCost = objFunc.PrintCost(w, ZT, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w - objFunc.optSolution).^2);

tstart = tic;

for t = 1:iterNum % for each iteration
    w = w - eta * (objFunc.Gradient(w, Z, ZT) + objFunc.lambda * w);

    % print cost
    if mod(t, 2) == 0
        cost = objFunc.PrintCost(w, ZT, t/2);
        if cost <= objFunc.optCost
            fprintf('Oops, we attain the optimal solution ...\n');
        else
            error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
            distance = sum((w - objFunc.optSolution).^2) / initDistance;
            subOptimality = [subOptimality; [t/2, toc(tstart), error, distance]];
        end
    end
    now = toc(tstart);
    if now > ourlimit
        break;
    end
end % iteration

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
% fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

label = 'GD';
curve_style = 'b-.';
% PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
