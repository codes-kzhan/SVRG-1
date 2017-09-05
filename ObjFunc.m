classdef ObjFunc
    properties
        lambda = 0;
        L = 0;
        mu = 0;
        optSolution;
        optCost;
    end
    methods
        % constructor
        function obj = ObjFunc(lambda, L, mu)
            obj.lambda = lambda;
            obj.L = L;
            obj.mu = mu;
        end

        % compute cost
        function costValue = Cost(obj, w, X, y)
            loss = mean(log(1 + exp(-y .* ((w'*X)'))));
            regularizer = obj.lambda / 2 * sum(w.^2);
            costValue = loss + regularizer;
        end

        % compute gradient without regularizer
        function gradient = Gradient(~, w, X, y))
            tmpExp = exp(-y .* (w'*X)')'; % 1-by-n vector
            gradient = mean((-y' .* tmpExp' .* X ) ./ (1 + tmpExp), 2);
        end

        % compute hypothesis
        function hypothesis = Hypothesis(~, w, X)
            tmpH = exp(w'/2 * X)';
            denominator = tmpH + 1 ./ tmpH;
            hypothesis = tmpH ./ denominator;
        end

        % print cost
        function cost = PrintCost(obj, w, X, y, stage)
            cost = obj.Cost(w, X, y);
            fprintf('epoch: %4d, cost: %.50f\n', stage, cost);
        end

        % predict
        function labels = Predict(obj, w, X)
            [~ , n] = size(X);
            y = obj.Hypothesis(w, X);
            labels = ones(n, 1);
            labels(y < 0.5) = -1;
        end

        % score
        function score = Score(obj, w, X, y)
            [~ , n] = size(X);
            labels = obj.Predict(w, X);
            score = sum(labels == y)/n;
        end
    end
end
