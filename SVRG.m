function wOpt = SVRG(objFunc, X, y, passes, factor)

fprintf('Fitting data with SVRG ...\n');

% initialization
[n ,d] = size(X);
iterNum = n;
eta = factor / objFunc.L
wtilde = zeros(d, 1);
w = wtilde;

for s = 1:passes % for each epoch
    ntilde = objFunc.Gradient(wtilde, X, y);
    objFunc.PrintCost(wtilde, X, y, s - 1);

    for i = 1:iterNum
        idx = randperm(n, 1);
        wDelta = objFunc.Gradient(w, X(idx, :), y(idx)) - objFunc.Gradient(wtilde, X(idx, :), y(idx)) + objFunc.lambda * w + ntilde;
        w = w - eta * wDelta;
    end
    wtilde = w;
end % epoch

objFunc.PrintCost(wtilde, X, y, s);
wOpt = wtilde;

end  % function
