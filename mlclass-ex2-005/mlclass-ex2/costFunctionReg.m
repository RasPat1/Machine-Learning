function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
A = -1 .* y .* log(h) .- (1 .- y) .* log(1 .- h);
J = sum(A) / m;


grad = (h .- y)' * X ./ m;

% Theta with the first entry replaced by 0
regTheta = theta;
regTheta(1) = 0;

regJ = lambda / (2 * m) * sum(regTheta .^ 2);
J = J + regJ;

gradReg = lambda / m .* regTheta';
grad = grad + gradReg;


% =============================================================

end
