function wOpt = FindOptSolution(objFunc, X, y, Xtest, ytest, passes, factor)

tstart = tic;
fprintf('Computing optimal solution ...\n');

% initialization
[d ,n] = size(X);
iterNum = n;

eta = factor / objFunc.L
% eta = 5e-1

wtilde = zeros(d, 1);
w = wtilde;
wOpt = wtilde;
optCost = objFunc.PrintCost(wtilde, X, y, 0);;
preCost = optCost;
reward = 0;


for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);

    for i = 1:iterNum
        idx = randperm(n, 1);
        wDelta = objFunc.Gradient(w, X(:, idx), y(idx)) - objFunc.Gradient(wtilde, X(:, idx), y(idx)) + objFunc.lambda * w + ntilde;
        w = w - eta * wDelta;
    end
    wtilde = w;
    currentCost = objFunc.PrintCost(wtilde, X, y, s);
    if currentCost <= optCost
        optCost = currentCost;
        wOpt = w;
    end
    if currentCost == preCost
        reward = reward + 1;
        if reward >= 10
            break;
        end
    else
        reward = 0;
    end
    preCost = currentCost;

end % epoch

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

end  % function
