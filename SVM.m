classdef SVM
    properties
        lambda = 0;
        L = 0;
        mu = 0;
        optSolution;
        optCost;
    end
    methods
        % constructor
        function obj = SVM(lambda, L, mu)
            obj.lambda = lambda;
            obj.L = L;
            obj.mu = mu;
        end

        % compute cost
        % Z = -y.*X, where y: N-by-1, X:d-by-N
        function costValue = Cost(obj, w, ZT)
            loss = mean(max(1 + ZT * w, 0).^2);
            regularizer = obj.lambda / 2 * sum(w.^2);
            costValue = loss + regularizer;
        end

        % compute gradient without regularizer
        function gradient = Gradient(~, w, Z, ZT)
            [d, n] = size(Z);
            gradient = Z * max(1 + ZT * w, 0) * 2/n;
        end

        % compute hypothesis
        function hypothesis = Hypothesis(~, w, X)
            hypothesis = X' * w;
        end

        % print cost
        function cost = PrintCost(obj, w, ZT, stage)
            cost = obj.Cost(w, ZT);
            fprintf('epoch: %4d, cost: %.50f\n', stage, cost);
        end

        % predict
        function labels = Predict(obj, w, X)
            [~ , n] = size(X);
            y = obj.Hypothesis(w, X);
            labels = ones(n, 1);
            labels(y < 0) = -1;
        end

        % score
        function score = Score(obj, w, X, y)
            [~ , n] = size(X);
            labels = obj.Predict(w, X);
            score = sum(labels == y)/n;
        end
    end
end
