function wOpt = SAGA(objFunc, X, y, Xtest, ytest, passes, factor)

tstart = tic;
fprintf('Fitting data with SAGA ...\n');

% initialization
[d ,n] = size(X);
iterNum = n * passes;

if issparse(X)
    w = sparse(d, 1);
else
    w = zeros(d, 1);
end

eta = factor/objFunc.L

subOptimality = zeros(passes, 1);
validPoints = 0;

initCost = objFunc.PrintCost(w, X, y, 0);
validPoints = validPoints + 1;
subOptimality(1) = 0;

tmpExp = exp(-y .* (w' * X)')';
gradients = ((-y' .* tmpExp) ./ (1 + tmpExp) .* X) + objFunc.lambda * w;  % d-by-n matrix

sumIG = sum(gradients, 2);

for t = 1:iterNum % for each iteration
    % update w
    idx = randperm(n, 1);
    newGrad = objFunc.Gradient(w, X(:, idx), y(idx)) + objFunc.lambda * w;
    oldGrad = gradients(:, idx);
    w = w - eta * (newGrad - oldGrad + sumIG/n);

    % update what we store
    sumIG = sumIG - oldGrad + newGrad;
    gradients(:, idx) = newGrad;

    % print and plot
    if mod(t, n) == 0
        cost = objFunc.PrintCost(w, X, y, round((t - 1)/n));
        if cost <= objFunc.optCost
            fprintf('Oops, we attain the optimal solution ...\n');
        else
            validPoints = validPoints + 1;
            subOptimality(validPoints) = log10((cost - objFunc.optCost)/(initCost - objFunc.optCost));
        end

    end
end % iteration

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

label = 'SAGA';
curve_style = 'g-';
PlotCurve(0:validPoints-1, subOptimality(1:validPoints), curve_style, label);

end  % function
