clear;
clc;

%% 8919基因 5080病

%% 加载数据
load('ppi_network.mat');
load('g_p_network.mat');
load('phenotype_network.mat');

%% 获取基因数量
gene_num = size(ppi_network, 1);

%% W是基因之间的关系
W = ppi_network;
D = diag(sum(ppi_network)); % D只有(i,i)位置有值，行求和 

%% 对W求值，根据论文第6页
for i = 1:gene_num
    for j = 1:gene_num
        W(i,j) = W(i,j)/sqrt(D(i,i)*D(j,j)); 
    end
end

%% 疾病之间的关系
phen = phenotype_network(:,2:5081);

%% 找到给出疾病的index
for i = 1:5080 
    if strcmp(phenotype_name{i},'PFEIFFER SYNDROME') == 1
        index = i; 
        break;
    end
end
% 疾病名
query_phenotype = phenotype_name{index}; 

% 取出该疾病对应的行
phenotype_array = phen(index,:);
temp = sort(phenotype_array);  % 排序

%% 找到相似度最高的15个病
similar_phenotype = find(phenotype_array > temp(5064));
% 去除自己本身
similar_phenotype(similar_phenotype == index)=[]; 

% 找到求和大于0的列，即和基因相关的病
phe_related_gene = find(sum(g_p_network,1)>0);
% 求交集
phenotype_index = intersect(similar_phenotype, phe_related_gene);

% 找到对应基因的下标
for i = 1:size(phenotype_index,2)
    gene_index(i) = find( g_p_network(:,phenotype_index(i))>0 );
end

%% 代入算法计算
F = zeros(gene_num,1); 
F(gene_index) = 1; %初始化
Y = F; 
t = 50; % 迭代次数
alpha = 0.9;
for i = 1:t
    F = alpha * (W * F) + (1-alpha) * Y; %论文第6页公式
end

[m,p] = max(F); % 取最大相关的基因下标到p
gene_name{p}