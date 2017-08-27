classdef ObjFunc
    properties
        lambda = 0;
    end
    methods
        % compute cost
        function costValue = Cost(obj, w, X, y)
            loss = mean(log(1 + exp(-y .* X*w)));
            regularizer = lambda / 2 * sum(w.^2);
            costValue = loss + regularizer;
        end

        % compute gradient without regularizer
        function gradient = Gradient(obj, w, X, y)
            tmpExp = exp(-y .* X*w);
            gradient = mean((-y .* tmpExp .* X ) ./ (1 + tmpExp));
        end

        % compute hypothesis
        function hypothesis = Hypothesis(obj, w, X)
            tmpH = exp(X * w/2);
            denominator = tmpH + 1/tmpH;
            hypothesis = tmpH / denominator;
        end

        % print cost
        function PrintCost(obj, w, X, y, stage)
            % @TODO
        end
    end
end
