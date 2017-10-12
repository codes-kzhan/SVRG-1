function wOpt = FindOptSolution(objFunc, X, y, Z, ZT, Xtest, ytest, passes, factor, batchSize)

tstart = tic;
fprintf('Computing optimal solution ...\n');

% initialization
[d ,n] = size(X);

iterNum = round(n/batchSize);
lambda = objFunc.lambda;

eta = factor / objFunc.L
% eta = 5e-1

if issparse(X)
    wtilde = sparse(d, 1);
else
    wtilde = zeros(d, 1);
end
w = wtilde;
wOpt = wtilde;
optCost = objFunc.PrintCost(wtilde, ZT, 0);;
preCost = optCost;
reward = 0;


for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, Z, ZT);

    for i = 1:iterNum
        idx = randperm(n, batchSize);
        Ztmp = Z(:, idx);
        ZTtmp = Ztmp';

        tmpDeltaG = Ztmp * (max(1 + ZTtmp * w, 0) - max(1 + ZTtmp * wtilde, 0)) * 2/batchSize;

        wDelta1 = tmpDeltaG + lambda * w;
        wDelta2 = wDelta1 + ntilde;
        w = w - eta * wDelta2;
    end
    wtilde = w;
    currentCost = objFunc.PrintCost(wtilde, ZT, s);
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
% fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);

end  % function
