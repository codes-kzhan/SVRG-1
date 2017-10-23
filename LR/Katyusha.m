function subOptimality = Katyusha(objFunc, X, y, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit)

fprintf('Fitting data with Katyusha...\n');

% initialization
[d ,n] = size(X);
iterNum = floor(n/batchSize);
done = iterNum*batchSize;  % this variable tells us whether we need to do the last iteration
lambda = objFunc.lambda;

eta = factor / objFunc.L
% eta = 5e-1

% if issparse(X)
%     wtilde = sparse(d, 1);
% else
%     wtilde = zeros(d, 1);
% end
wtilde = zeros(d, 1);
w = zeros(d, 1);
u = zeros(d, 1);
z = zeros(d, 1);


initCost = objFunc.PrintCost(wtilde, X, y, 0);
subOptimality = [0, 0, 1, 1];
objOptNorm = sum(objFunc.optSolution.^2);

tstart = tic;

tau2 = 1/2;
% tau1 = min(sqrt(iterNum * objFunc.mu / 3 / objFunc.L), 1/2);
% alpha = 1/(3 * tau1 * objFunc.L);

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);
    ntilde = full(ntilde);
    tau1 = 1/(s+2);
    alpha = factor/(3 * tau1 * objFunc.L);

    for i = 1:iterNum
        % idx = mod(i-1, n) + 1;
        % idx = randperm(n, batchSize);
        % w = tau1 * z + tau2 * wtilde + (1 - tau2 - tau1) * u;
        %
        % Xtmp = X(:, idx);
        % ytmp = y(idx);
        % % new gradient
        % tmpExp = exp(-ytmp .* (Xtmp' *w))'; % 1-by-n vector
        % % old gradient
        % tmpExpTilde = exp(-ytmp .* (Xtmp' * wtilde))'; % 1-by-n vector
        % wDelta1 = mean(-ytmp' .* (1./(1 + tmpExpTilde) - 1./(1 + tmpExp)) .* Xtmp, 2);
        %
        % wDelta2 = wDelta1 + lambda * w;
        % wDelta = wDelta2 + ntilde;
        %
        % znew = z - alpha * wDelta;
        % u = w + tau1 * (znew - z);
        % z = znew;
        Katyusha_logistic(w, wtilde, ntilde, X, y, lambda, eta, iterNum, u, z, tau1, tau2);

    end
    wtilde = u;

    % print and plot
    cost = objFunc.PrintCost(wtilde, X, y, s);
    if cost <= objFunc.optCost
        fprintf('Oops, we attain the optimal solution ...\n');
    else
        error = (cost - objFunc.optCost)/(initCost - objFunc.optCost);
        distance = sum((wtilde - objFunc.optSolution).^2) / objOptNorm;
        subOptimality = [subOptimality; [s, toc(tstart), error, distance]];
    end
    now = toc(tstart);
    if now > ourlimit
        break;
    end
end % epoch

wOpt = wtilde;

telapsed = toc(tstart);
fprintf('training accuracy: %f\n', objFunc.Score(wOpt, X, y));
fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'Katyusha';
curve_style = 'b-.';
PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
