function subOptimality = DIG(objFunc, X, y, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit)

fprintf('Fitting data with DIG ...\n');

% initialization
[d ,n] = size(X);
iterNum = floor(2 * n / batchSize);


% if issparse(X)
%     w = sparse(d, 1);
% else
%     w = zeros(d, 1);
% end
w = zeros(d, 1);

initCost = objFunc.PrintCost(w, X, y, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w- objFunc.optSolution).^2);
lambda = objFunc.lambda;

tstart = tic;

for s = 1:passes % for each iteration
    % idx = mod(t-1, n) + 1;
    % idx = mod((t-1) * batchSize, n) + 1 : mod(t * batchSize - 1, n) + 1;
    % Xtmp = X(:, idx);
    % ytmp = y(idx);
    %
    % % eta = min(2/(objFunc.lambda * (t + 1)), 1e-2);
    DIG_logistic(w, X, y, lambda, s, iterNum, factor);
    %
    % tmpExp = exp(-ytmp .* (Xtmp' *w))'; % 1-by-n vector
    % newGrad = mean(((-ytmp' .* tmpExp) ./ (1 + tmpExp) .* Xtmp) + objFunc.lambda * w, 2);
    % w = w - eta * newGrad;

    cost = objFunc.PrintCost(w, X, y, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
        distance = sum((w- objFunc.optSolution).^2) / initDistance;
        subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
    end
    if error <= ourlimit
        break;
    end

end % iteration

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
% fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

label = 'DIG';
curve_style = ':';
% PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
