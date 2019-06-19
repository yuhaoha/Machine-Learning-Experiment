clear;close all;clc;

%% 加载数据
load('MovieLens.mat');
V = Y; %1682,943

[m,n] = size(V); %m是行 n是列
R = 10;  %10个分类
K = 200; %迭代次数
W = abs(rand(m,R)); % 电影 
H = abs(rand(R,n));  % 用户

%% 实现update
for i = 1:K
    H = H .* (W'*V) ./ ((W'*W)*H);  
    W = W .* (V*H') ./ (W*(H*H'));  
end

[n,m]=size(H);

s = zeros(m,m); % 943,943

for i=1:m
    for j=1:i
        s(i,j) = abs(corr2(H(:,i),H(:,j)));
        s(j,i)=s(i,j); % 对称矩阵
    end
end

%% 显示热力图
imshow(mat2gray(s));