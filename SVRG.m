[optW] = SVRG(X, y, passes, factor)

fprintf('SVRG');

% initialization
[n d] = size(X);
iterNum = n;
wtidle = zeros(d, 1);

for s = 1:passes % for each epoch

    PrintCost(); % @TODO

    w = wtidle;
    ntidle = Gradient(w, X, y)
    for i = 1:iterNum
        idx = randperm(n, 1);
    end

end % epoch

end  % function
