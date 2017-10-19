
function a = prox(x,sigma, param, lambda1)

switch param
case 1, a = sign(x) .* max( abs(x) - lambda1 * sigma, 0); % L1-norm, checked
end
