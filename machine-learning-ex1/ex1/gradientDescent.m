function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %��¼ÿ�ε����Ľ��

for iter = 1:num_iters
% ����1500��
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ����ʵ���ݶ��½� theta = [2,1]�������   X = [97,2]��ʾӰ�����ص�ֵ   y=[97,1]��ʾ���
    theta = theta - alpha * 1/m * (X' * (X*theta-y)); 



    % ============================================================

    % Save the cost J in every iteration    
    % ��ÿ�ε����Ľ�����浽J_history
    J_history(iter) = computeCost(X, y, theta);

end

end
