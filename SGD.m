function wOpt = SGD(objFunc, X, y, Xtest, ytest, passes, ~)

tstart = tic;
fprintf('Fitting data with SGD ...\n');

% initialization
[n ,d] = size(X);
iterNum = n * passes;
batchSize = 64;
w = zeros(d, 1);
% wOpt = 0;

subOptimality = zeros(passes, 1);
validPoints = 0;

initCost = objFunc.PrintCost(w, X, y, 0);
validPoints = validPoints + 1;
subOptimality(1) = 0;

for t = 1:iterNum % for each iteration
    idx = randperm(n, batchSize);
    eta = min(2/(objFunc.lambda * (t + 1)), 1e-2);
    w = w - eta * (objFunc.Gradient(w, X(idx, :), y(idx)) + objFunc.lambda * w);
    % wOpt = wOpt + 2 * t * w / (iterNum * (iterNum + 1));

    % print and plot
    if mod(t, n) == 0
        cost = objFunc.PrintCost(w, X, y, round((t - 1)/n));
        if cost <= objFunc.optCost
            fprintf('Oops, we attain the optimal solution ...');
        else
            validPoints = validPoints + 1;
            subOptimality(validPoints) = log((cost - objFunc.optCost)/(initCost - objFunc.optCost));
        end

    end
end % iteration

wOpt = w;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

label = 'SGD';
curve_style = 'b-';
PlotCurve(0:validPoints-1, subOptimality(1:validPoints), curve_style, label);

end  % function
