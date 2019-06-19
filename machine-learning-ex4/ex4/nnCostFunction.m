function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               %400 25 10 5000*400 5000*1 0��1����)
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
Theta1_grad = zeros(size(Theta1));  %��theta1ά����ͬ��ȫΪ0 [25,401]
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

    
% Y��ά�� [5000,10]����i�еĵ�j��Ϊ1��jΪ��������ֵ
% ��y��[5000,1]���[5000,10]
Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
end

% Theta1ά�� = [25, 401], Theta2ά�� = [10, 26]
% y��Ԥ��ֵ y_predά�� = [5000, 10];
% ��ǰ������y_pred ���󻯴���ѭ����ѭ��ÿ�εõ�[10,1]�������������ж�Ӧһ��
X = [ones(m, 1) X]; % Xά�� = [5000, 401]
a1 = X;
z2 = a1 * Theta1'; %z2ά�� = [5000,25]
a2 = sigmoid(z2);  
a2 = [ones(m, 1) a2]; % a2ά�� = [5000, 26];
z3 = a2 * Theta2';
a3 = sigmoid(z3); %z3,a3ά�� = [5000,10]
y_pred = a3;
%���ۺ������㣬ʹ�þ�����㣬��һ��sum�������(��Ӧk 1...10)���ڶ����������(��Ӧi 1...5000)
J = -1.0 / m * sum( sum( Y.*log(y_pred)+(1-Y).*log(1-y_pred) ) );

% ���򻯣���һ��Ϊƫ�õ�Ԫ������������ Theta1 = [25,401] Theta2 = [10,26]
J = J + lambda / (2.0*m) * ( sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2)) );

% ���򴫲�
delta3 = a3 - Y;  % ά�� = [5000, 10]
% ������ƫ�õ�Ԫ ([10,25]' * [5000,10]')'.*[5000,25] ÿ�����������ز�����
delta2 = (Theta2(:,2:end)'* delta3')'.*sigmoidGradient(z2);  % ά�� = [5000, 25]
% Delta1����һ��Եڶ������������Ϊ����ʱdelta��������a1��ˣ��˴���Ҫת��
%m������[25,1]*[1,401]���ۼ� �ȼ��ھ������
Delta1 = delta2'*a1;  % ά�� = [25, 400+1]����Ϊ�Ƕ�theta1������theta1ά����ͬ
Delta2 = delta3'*a2;  % ά�� = [10, 25+1] 
 
% �ݶ�
Theta1_grad = Delta1/m;  
Theta2_grad = Delta2/m;  

% ���� j=1ʱ��ƫ�õ�Ԫ����Ҫ����,����һ������Ϊ0
Theta1(:,1) = 0;  
Theta2(:,1) = 0;  
Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients ������
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
