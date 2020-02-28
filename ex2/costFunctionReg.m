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

theta_reg = theta(2:end,:);

hx = 1 ./ (1 + exp(-X*theta));

J = ((1/m)*sum(-y.*log(hx) - (1-y).*log(1-(hx)))) + (lambda/(2*m)*sum(theta_reg.^2));



%  for i = 1:size(theta,1),
%    X_1 = X(:,i);
 %   if i == 1
 %       grad(i) = (1/m)*sum((hx - y).*X_1);
  %  else
  %      grad(i) = (1/m)*sum((hx - y).*X_1) + (lambda/m)*theta(i);
  %  end;
  %end;


  theta_temp = [0; theta(2:end,:)];
  grad = (1/m)*(pinv(X)*(hx-y)) + (lambda/m)*theta_temp;
% =============================================================

end
