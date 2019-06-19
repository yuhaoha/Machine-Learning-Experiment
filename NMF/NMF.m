clear;close all;clc;

%% ��������
load('MovieLens.mat');
V = Y; %1682,943

[m,n] = size(V); %m���� n����
R = 10;  %10������
K = 200; %��������
W = abs(rand(m,R)); % ��Ӱ 
H = abs(rand(R,n));  % �û�

%% ʵ��update
for i = 1:K
    H = H .* (W'*V) ./ ((W'*W)*H);  
    W = W .* (V*H') ./ (W*(H*H'));  
end

[n,m]=size(H);

s = zeros(m,m); % 943,943

for i=1:m
    for j=1:i
        s(i,j) = abs(corr2(H(:,i),H(:,j)));
        s(j,i)=s(i,j); % �Գƾ���
    end
end

%% ��ʾ����ͼ
imshow(mat2gray(s));