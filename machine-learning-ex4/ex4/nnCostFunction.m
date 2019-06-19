function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               %400 25 10 5000*400 5000*1 0（1正则化)
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
                 hidden_layer_size, (input_layer_size + 1));  % [25,401]

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  %[10,26]

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));  %与theta1维度相同，全为0 [25,401]
Theta2_grad = zeros(size(Theta2));  % [10,26]

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

    
% Y的维度 [5000,10]，第i行的第j个为1，j为此样本的值
% 将y从[5000,1]变成[5000,10]
Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
end

% Theta1维度 = [25, 401], Theta2维度 = [10, 26]
% y的预测值 y_pred维度 = [5000, 10];
% 向前传播求y_pred 矩阵化代替循环，循环每次得到[10,1]列向量，矩阵中对应一行
X = [ones(m, 1) X]; % X维度 = [5000, 401]
a1 = X;
z2 = a1 * Theta1'; %z2维度 = [5000,25]
a2 = sigmoid(z2);  
a2 = [ones(m, 1) a2]; % a2维度 = [5000, 26];
z3 = a2 * Theta2';
a3 = sigmoid(z3); %z3,a3维度 = [5000,10]
y_pred = a3;
%代价函数计算，使用矩阵计算，第一个sum对列求和(对应k 1...10)，第二个对行求和(对应i 1...5000)
J = -1.0 / m * sum( sum( Y.*log(y_pred)+(1-Y).*log(1-y_pred) ) );

% 正则化，第一列为偏置单元，不参与正则化 Theta1 = [25,401] Theta2 = [10,26]
J = J + lambda / (2.0*m) * ( sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2)) );

% 反向传播
delta3 = a3 - Y;  % 维度 = [5000, 10]
% 不计算偏置单元 ([10,25]' * [5000,10]')'.*[5000,25] 每个样本对隐藏层的误差
delta2 = (Theta2(:,2:end)'* delta3')'.*sigmoidGradient(z2);  % 维度 = [5000, 25]
% Delta1：第一层对第二层的误差矩阵，因为迭代时delta行向量与a1相乘，此处需要转置
%m个样本[25,1]*[1,401]再累加 等价于矩阵相乘
Delta1 = delta2'*a1;  % 维度 = [25, 400+1]，因为是对theta1的误差，与theta1维度相同
Delta2 = delta3'*a2;  % 维度 = [10, 25+1] 
 
% 梯度
Theta1_grad = Delta1/m;  
Theta2_grad = Delta2/m;  

% 正则化 j=1时，偏置单元不需要正则化,将第一列设置为0
Theta1(:,1) = 0;  
Theta2(:,1) = 0;  
Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients 列向量
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
