function subOptimality = SVRGNR(objFunc, X, y, Xtest, ytest, passes, factor, batchSize, dataset, gridNum, ourlimit)

fprintf('Fitting data with SVRG-NR ...\n');

% initialization
[d, n] = size(X);
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

initCost = objFunc.PrintCost(wtilde, X, y, 0);
subOptimality = [0, 0, 1, 1];
objOptNorm = sum(objFunc.optSolution.^2);

tstart = tic;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);


    % for i = 1:iterNum
    %     idx = (i-1)*batchSize + 1 : i*batchSize;
    %     Xtmp = X(:, idx);
    %     ytmp = y(idx);
    %     % new gradient
    %     tmpExp = exp(-ytmp .* (Xtmp' *w))'; % 1-by-n vector
    %     % old gradient
    %     tmpExpTilde = exp(-ytmp .* (Xtmp' * wtilde))'; % 1-by-n vector
    %     wDelta1 = mean(-ytmp' .* (1./(1 + tmpExpTilde) - 1./(1 + tmpExp)) .* Xtmp, 2);
    %
    %     wDelta2 = wDelta1 + lambda * w;
    %     wDelta3 = wDelta2 + ntilde;
    %     w = w - eta * wDelta3;
    % end
    %
    % if done < n
    %     idx = done + 1 : n;
    %     Xtmp = X(:, idx);
    %     ytmp = y(idx);
    %     % new gradient
    %     tmpExp = exp(-ytmp .* (w'*Xtmp)')'; % 1-by-n vector
    %     % old gradient
    %     tmpExpTilde = exp(-ytmp .* (wtilde'*Xtmp)')'; % 1-by-n vector
    %     wDelta1 = mean(-ytmp' .* (tmpExp./(1 + tmpExp) - tmpExpTilde./(1 + tmpExpTilde)) .* Xtmp, 2);
    %
    %     wDelta2 = wDelta1 + lambda * w;
    %     wDelta3 = wDelta2 + ntilde;
    %     w = w - eta * wDelta3;
    % end
    % wtilde = w;
    ntilde = full(ntilde);
    SIG_logistic(w, wtilde, ntilde, X, y, lambda, eta, iterNum);
    wtilde(:) = w(:); % to avoid copy-on-write


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
% fprintf('test accuracy: %f\n', objFunc.Score(wOpt, Xtest, ytest));
fprintf('time elapsed: %f\n', telapsed);


label = 'DVRG';
curve_style = '-';

% PlotTime(subOptimality, curve_style, label, dataset, gridNum);
% PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
