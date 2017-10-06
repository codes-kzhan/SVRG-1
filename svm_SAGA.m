function subOptimality = svm_SAGARR(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, dataset, gridNum)

fprintf('Fitting data with SAGA ...\n');

% initialization
[d ,n] = size(X);
iterNum = n * passes;

eta = factor/objFunc.L

if issparse(X)
    w = sparse(d, 1);
else
    w = zeros(d, 1);
end

initCost = objFunc.PrintCost(w, X, y, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w- objFunc.optSolution).^2);

tstart = tic;

% tmpExp = exp(-y .* (X' * w))';
% gradients = ((-y' .* tmpExp) ./ (1 + tmpExp) .* X) + objFunc.lambda * w;  % d-by-n matrix
gradients = Z * (2 .* (max(1 + ZT * w, 0))') + objFunc.lambda * w;  % d-by-n matrix

sumIG = sum(gradients, 2);

for t = 1:iterNum % for each iteration
    % update w
    % idx = randperm(n, 1);
    idx = mod(t-1, n) + 1;
    Ztmp = Z(:, idx);
    ZTtmp = Ztmp';

    newGrad = Ztmp * (2 .* max(1 + ZTtmp * w, 0)) + objFunc.lambda * w;
    oldGrad = gradients(:, idx);
    w = w - eta * (newGrad - oldGrad + sumIG/n);

    % update what we store
    sumIG = sumIG - oldGrad + newGrad;
    gradients(:, idx) = newGrad;

    % print and plot
    if mod(t, n) == 0
        order = randperm(size(X, 2));
        X = X(:, order); % random shuffle
        gradients = gradients(:, order); % random shuffle
        y = y(order); % random shuffle

        s = round(t/n);
        cost = objFunc.PrintCost(w, X, y, s);
        if cost <= objFunc.optCost
            fprintf('Oops, we attain the optimal solution ...\n');
        else
            error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
            distance = sum((w- objFunc.optSolution).^2) / initDistance;
            subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
        end

    end
end % iteration

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

label = 'SAGA-RR';
curve_style = 'g-';
PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
