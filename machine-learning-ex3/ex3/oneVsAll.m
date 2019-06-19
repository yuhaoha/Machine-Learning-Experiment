function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);  %行 数据数量 5000
n = size(X, 2);  %列 theta个数 400

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1); %[10,401] 每一行对应一个数字的Logistic回归 1 2 3...0

% Add ones to the X data matrix
X = [ones(m, 1) X]; % X = [5000,401]

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

for c = 1:num_labels %从1到10循环计算每个数字对应的theta
    initial_theta = zeros(n + 1, 1); % initial_theta = [401,1]   theta维度也是[401,1]
    options = optimset('GradObj', 'on', 'MaxIter', 50); % 迭代50次
    [theta] = ...  %(y == c)返回一个[5000,1]的0,1向量，来执行本次Logistic回归(y==c为1，否则为0)
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ... %lrCostFunction计算单变量logistic回归，
                 initial_theta, options);
    all_theta(c, :) = theta(:)';%[1,401]
end










% =========================================================================


end
