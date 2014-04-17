function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add input bias to X
X = [ones(size(X, 1), 1) X];

z2 = Theta1 * X';
a2 = sigmoid(z2);
z3 = Theta2 * [ones(size(a2, 2), 1) a2']';
a3 = sigmoid(z3);
h = a3;
yJ = [y==1];
for i = 2:size(h, 1)
	yJ = [yJ y==i];
end;

% Loop version

% Jinner = [];
% for i = 1:size(X, 1)
% 	yT = yJ(i, :);
% 	hT = h(:, i);
% 	term1 = -1 * yT * log(hT);
% 	term2 = (1 - yT) * log(1 - hT);
% 	inner = term1 - term2;
% 	Jinner(i) = sum(inner);
% end;

% J = sum(Jinner) / m;



term1 = -1 * yJ * log(h);
term2 = (1 - yJ) * log(1 - h);
inner = term1 - term2;

% trace rather than sum
% inner has every training example graded against every result
% We only want ex1 vs result 1 and 2 v 2 so take the diagnoal only. 

J = trace(inner) / m;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Return Theta1_grad and Theta2_grad


% yJ is y converted to column vectors for each expected result

D_1 = 0;
D_2 = 0;

for t = 1:m
% 	X is already modified with a bias layer just pull it out
	a_1 = X(t, 1:end)';
	% size(a_1)	% 401 x 1

	z_2 = Theta1 * a_1;
	a_2 = sigmoid(z_2);
	z_3 = Theta2 *  [1; a_2];
	a_3 = sigmoid(z_3);

	yt = yJ(t, :)';
	% size(yt)	% 10 x 1

	d_3 = a_3 - yt;
	% size(d_3)	% 10 x 1


	d_2 = Theta2' * d_3 .* sigmoidGradient([1; z_2;]);
	% size(d_2) 	% 26 x 1

	d_2 = d_2(2:end);

	D_1 = D_1 + d_2 * a_1';
	D_2 = D_2 + d_3 * [1; a_2]';
end

Theta1_grad = D_1 ./ m;		
Theta2_grad = D_2 ./ m;		

% size(Theta1_grad)	% 5 x 4
% size(Theta2_grad)	% 3 x 5


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% exclude bias terms
T1Reg = Theta1(1:end, 2:end);
T2Reg = Theta2(1:end, 2:end);

% unroll it all
TReg = [T1Reg(:); T2Reg(:);];

regParam = lambda / (2 * m);
regTerm = regParam * sum(TReg .^ 2);

J = J + regTerm;

% For gradients





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
