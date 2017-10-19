function wOpt = FindOptSolution(objFunc, X, y, Xtest, ytest, passes, factor, batchSize)

tstart = tic;
fprintf('Computing optimal solution ...\n');

% initialization
[d ,n] = size(X);

iterNum = round(n/batchSize);
lambda2 = objFunc.lambda2;

eta = factor / objFunc.L
% eta = 5e-1

if issparse(X)
    wtilde = sparse(d, 1);
else
    wtilde = zeros(d, 1);
end
w = wtilde;
wOpt = wtilde;
optCost = objFunc.PrintCost(wtilde, X, y, 0);;
preCost = optCost;
reward = 0;


for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);

    for i = 1:iterNum
        idx = randperm(n, batchSize);
        Xtmp = X(:, idx);
        ytmp = y(idx);
        % new gradient
        tmpExp = exp(-ytmp .* (Xtmp' *w))'; % 1-by-n vector
        % old gradient
        tmpExpTilde = exp(-ytmp .* (Xtmp' * wtilde))'; % 1-by-n vector
        wDelta1 = mean(-ytmp' .* (1./(1 + tmpExpTilde) - 1./(1 + tmpExp)) .* Xtmp, 2);

        wDelta2 = wDelta1 + lambda2* w;
        wDelta3 = wDelta2 + ntilde;
        w = prox(w - eta*wDelta3, eta, 1, objFunc.lambda1);
    end
    wtilde = w;
    currentCost = objFunc.PrintCost(wtilde, X, y, s);
    if currentCost <= optCost
        optCost = currentCost;
        wOpt = w;
        save('wOpt_tmp.mat', 'wOpt');
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
