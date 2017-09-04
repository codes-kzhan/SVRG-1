function wOpt = KatyushaNR(objFunc, X, y, Xtest, ytest, passes, factor)

tstart = tic;
fprintf('Fitting data with KatyushaNR...\n');

% initialization
[n ,d] = size(X);
iterNum = n * 2;
subOptimality = zeros(passes, 1);
validPoints = 0;

eta = factor / objFunc.L
% eta = 5e-1

wtilde = zeros(d, 1);


initCost = objFunc.PrintCost(wtilde, X, y, 0);
validPoints = validPoints + 1;
subOptimality(1) = 0;

tau2 = 1/2;
tau1 = min(sqrt(iterNum * objFunc.mu / 3 / objFunc.L), 1/2);
alpha = 1/(3 * tau1 * objFunc.L);
u = wtilde;
z = wtilde;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);

    for i = 1:iterNum
        idx = mod(i-1, n) + 1;
        % idx = randperm(n, 1);
        w = tau1 * z + tau2 * wtilde + (1 - tau2 - tau1) * u;
        wDelta = objFunc.Gradient(w, X(idx, :), y(idx)) - objFunc.Gradient(wtilde, X(idx, :), y(idx)) + objFunc.lambda * w + ntilde;
        znew = z - alpha * wDelta;
        u = w + tau1 * (znew - z);
        z = znew;
    end
    wtilde = u;

    % print and plot
    cost = objFunc.PrintCost(wtilde, X, y, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        validPoints = validPoints + 1;
        subOptimality(validPoints) = log10((cost - objFunc.optCost)/(initCost - objFunc.optCost));
    end
end % epoch

wOpt = wtilde;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'KatyushaNR';
curve_style = '-.';
PlotCurve(0:validPoints-1, subOptimality(1:validPoints), curve_style, label);

end  % function
