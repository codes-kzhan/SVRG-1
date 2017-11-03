function subOptimality = svm_SAGARR(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, dataset, gridNum, ourlimit)

fprintf('Fitting data with SAGA ...\n');

% initialization
[d ,n] = size(X);
iterNum = 2 * n;

eta = factor/objFunc.L

% if issparse(X)
%     w = sparse(d, 1);
% else
%     w = zeros(d, 1);
% end
w = zeros(d, 1);

initCost = objFunc.PrintCost(w, ZT, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w- objFunc.optSolution).^2);

tstart = tic;

sumIG = zeros(d,1);
gradients = zeros(n,1);
lambda = objFunc.lambda;

for s = 1:passes % for each iteration
    % % update w
    % % idx = randperm(n, 1);
    % idx = mod(t-1, n) + 1;
    % Ztmp = Z(:, idx);
    % ZTtmp = Ztmp';
    %
    % newGrad = Ztmp * (2 .* max(1 + ZTtmp * w, 0)) + objFunc.lambda * w;
    % oldGrad = gradients(:, idx);
    % w = w - eta * (newGrad - oldGrad + sumIG/n);
    %
    % % update what we store
    % sumIG = sumIG - oldGrad + newGrad;
    % gradients(:, idx) = newGrad;

    iVals = int32([randperm(n), randperm(n)]);
    SAGA_svm(w, Z, lambda, eta, sumIG, gradients, iterNum, iVals);

    % print and plot
    cost = objFunc.PrintCost(w, ZT, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
        distance = sum((w- objFunc.optSolution).^2) / initDistance;
        subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
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

label = 'SAGA-RR';
curve_style = '-';
% PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
