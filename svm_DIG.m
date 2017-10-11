function subOptimality = svm_DIG(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, dataset, gridNum)

fprintf('Fitting data with DIG...\n');

% initialization
[d ,n] = size(X);
iterNum = 2 * n * passes;

% eta = factor/objFunc.L

if issparse(X)
    w = sparse(d, 1);
else
    w = zeros(d, 1);
end
% w = zeros(d, 1);

initCost = objFunc.PrintCost(w, ZT, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w- objFunc.optSolution).^2);

tstart = tic;


for t = 1:iterNum % for each iteration

    eta = factor/((t/n + 1) + 1);
    % update w
    % idx = randperm(n, 1);
    idx = mod(t-1, n) + 1;
    Ztmp = Z(:, idx);
    ZTtmp = Ztmp';

    newGrad = Ztmp * (2 .* max(1 + ZTtmp * w, 0)) + objFunc.lambda * w;
    w = w - eta * newGrad;

    % print and plot
    if mod(t, n) == 0
      % order = randperm(size(X, 2));
      % Z = Z(:, order); % random shuffle
      % gradients = gradients(:, order); % random shuffle

        if mod(t, 2*n) == 0
            s = round(t/n/2);
            cost = objFunc.PrintCost(w, ZT, s);
            if cost <= objFunc.optCost
                fprintf('Oops, we attain the optimal solution ...\n');
            else
                error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
                distance = sum((w- objFunc.optSolution).^2) / initDistance;
                subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
            end
        end

    end
end % iteration

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

label = 'DIG';
curve_style = 'g-';
PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
