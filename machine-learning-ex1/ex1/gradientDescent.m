function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %记录每次迭代的结果

for iter = 1:num_iters
% 迭代1500次
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % 迭代实现梯度下降 theta = [2,1]代表参数   X = [97,2]表示影响因素的值   y=[97,1]表示结果
    theta = theta - alpha * 1/m * (X' * (X*theta-y)); 



    % ============================================================

    % Save the cost J in every iteration    
    % 将每次迭代的结果保存到J_history
    J_history(iter) = computeCost(X, y, theta);

end

end
