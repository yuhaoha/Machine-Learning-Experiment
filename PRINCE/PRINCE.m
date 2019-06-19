clear;
clc;

%% 8919���� 5080��

%% ��������
load('ppi_network.mat');
load('g_p_network.mat');
load('phenotype_network.mat');

%% ��ȡ��������
gene_num = size(ppi_network, 1);

%% W�ǻ���֮��Ĺ�ϵ
W = ppi_network;
D = diag(sum(ppi_network)); % Dֻ��(i,i)λ����ֵ������� 

%% ��W��ֵ���������ĵ�6ҳ
for i = 1:gene_num
    for j = 1:gene_num
        W(i,j) = W(i,j)/sqrt(D(i,i)*D(j,j)); 
    end
end

%% ����֮��Ĺ�ϵ
phen = phenotype_network(:,2:5081);

%% �ҵ�����������index
for i = 1:5080 
    if strcmp(phenotype_name{i},'PFEIFFER SYNDROME') == 1
        index = i; 
        break;
    end
end
% ������
query_phenotype = phenotype_name{index}; 

% ȡ���ü�����Ӧ����
phenotype_array = phen(index,:);
temp = sort(phenotype_array);  % ����

%% �ҵ����ƶ���ߵ�15����
similar_phenotype = find(phenotype_array > temp(5064));
% ȥ���Լ�����
similar_phenotype(similar_phenotype == index)=[]; 

% �ҵ���ʹ���0���У����ͻ�����صĲ�
phe_related_gene = find(sum(g_p_network,1)>0);
% �󽻼�
phenotype_index = intersect(similar_phenotype, phe_related_gene);

% �ҵ���Ӧ������±�
for i = 1:size(phenotype_index,2)
    gene_index(i) = find( g_p_network(:,phenotype_index(i))>0 );
end

%% �����㷨����
F = zeros(gene_num,1); 
F(gene_index) = 1; %��ʼ��
Y = F; 
t = 50; % ��������
alpha = 0.9;
for i = 1:t
    F = alpha * (W * F) + (1-alpha) * Y; %���ĵ�6ҳ��ʽ
end

[m,p] = max(F); % ȡ�����صĻ����±굽p
gene_name{p}