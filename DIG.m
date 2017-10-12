function subOptimality = DIG(objFunc, X, y, Xtest, ytest, passes, factor, batchSize, dataset, gridNum)

fprintf('Fitting data with DIG ...\n');

% initialization
[d ,n] = size(X);
iterNum = floor(2 * n * passes / batchSize);


% if issparse(X)
%     w = sparse(d, 1);
% else
%     w = zeros(d, 1);
% end
w = zeros(d, 1);
% w = zeros(d, 1);

initCost = objFunc.PrintCost(w, X, y, 0);
subOptimality = [0, 0, 1, 1];

initDistance = sum((w- objFunc.optSolution).^2);

tstart = tic;

for t = 1:iterNum % for each iteration
    % update w
    % idx = randperm(n, 1);
    % idx = mod(t-1, n) + 1;
    idx = mod((t-1) * batchSize, n) + 1 : mod(t * batchSize - 1, n) + 1;
    Xtmp = X(:, idx);
    ytmp = y(idx);

    % eta = min(2/(objFunc.lambda * (t + 1)), 1e-2);
    eta = factor/((floor(t * batchSize /2 * n) + 1) + 1);

    tmpExp = exp(-ytmp .* (Xtmp' *w))'; % 1-by-n vector
    newGrad = mean(((-ytmp' .* tmpExp) ./ (1 + tmpExp) .* Xtmp) + objFunc.lambda * w, 2);
    w = w - eta * newGrad;

    % print and plot
        % order = randperm(size(X, 2));
        % X = X(:, order); % random shuffle
        % gradients = gradients(:, order); % random shuffle
        % y = y(order); % random shuffle

    % if mod(t * batchSize, 2 * n) <= 2 * n && mod((t + 1) * batchSize, 2*n) >= 2 * n
    if mod(t * batchSize, 2 * n - mod(2 * n, batchSize)) == 0
        s = floor(t*batchSize/n/2);
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

label = 'DIG';
curve_style = ':';
PlotCurve(subOptimality, curve_style, label, dataset, gridNum);

end  % function
