%% 步骤 1: 基础设置与数据加载
clc; clear; 

% 路径配置
base_path = 'F:\Seafile\Research\Nonlocal\New_Mueller\Experiment\20251222_2\';
addpath(genpath(base_path)); 

% 加载系统矩阵
load([base_path, 'S_Encoder.mat']);
load([base_path, 'M_Decoder.mat']);
spec = load([base_path, 'Alluse\spec_grating_source.mat']).Spec1;
load([base_path, 'Alluse\Ms_grating_tra_all2.mat']); % 加载 Ms (7维数据)
% size(Ms)

% 重组 T 矩阵
fprintf('正在初始化 T 矩阵...\n');
T = zeros(256, 256, 121, 4, 4);
for i = 1:121
    for j = 1:4
        for k =1:4
            T(:,:,i,j,k) = M(:,:,i,j) .* S(:,:,i,k);
        end
    end
end
T1 = zeros(256, 256, 1936);
for j = 1:4
    for k =1:4
        for i = 1 :121
            T1(:,:,(j-1)*4*121+(k-1)*121+i) = T(:,:,i,j,k);
        end
    end
end
T1_reshaped = reshape(T1, [], 1936); 

%% 步骤 2: Mueller 矩阵库预处理
fprintf('正在预处理 Ms 库...\n');

range_i = 1:21;   
range_j = 1:21;   
range_k = 1:11;   
range_al = 1:7;   

Ni = length(range_i); Nj = length(range_j); Nk = length(range_k);
Nal = length(range_al); 
Total_Inner = Ni * Nj * Nk; % 总参数组合数

% 应用光谱
for i = 1:4 
    for j = 1:4
        Ms(i,j,:,:,:,:,:) = Ms(i,j,:,:,:,:,:) .* reshape(spec, [1, 1, 121, 1, 1, 1, 1]);
    end
end

% 构建 Ms1 (1936 x Ni x Nj x Nk x Nal)
Ms1 = zeros(1936, Ni, Nj, Nk, Nal);
for n4 = 1:Nal
    for n3 = 1:Nk
        for n2 = 1:Nj
            for n1 = 1:Ni
                temp_block = Ms(:,:,:,n1,n2,n3,n4); 
                temp_permuted = permute(temp_block, [3, 2, 1]); 
                Ms1(:, n1, n2, n3, n4) = temp_permuted(:);
            end
        end
    end
end

%% 步骤 3: 实验图像加载与 数据提取
fprintf('加载实验图像并提取向量...\n');

I1 = imread([base_path, 'phi0.png']); 
bk = imread([base_path, '10000\bk.png']);

I1 = Mean_Intensity_2dim(I1, 20);
bk = Mean_Intensity_2dim(bk, 20);
I1 = double(I1 - bk);

% 生成 Mask
temp_img = ones(size(I1));
masked_template = Select_circle(temp_img, 128, 128, 128); 
valid_idx = find(masked_template ~= 0); 

% 提取有效数据向量 (归一化)
I1_vec = I1(valid_idx);
max_I1 = max(I1_vec);
if max_I1 == 0, max_I1 = 1; end
I1_vec = I1_vec / max_I1; 

T1_valid = T1_reshaped(valid_idx, :); 

%% 步骤 4: 模拟强度计算 (替代原来的误差循环)
% 注意：这里不再计算误差，而是将归一化后的强度保存下来
fprintf('正在计算并保存模拟强度 (Intensity)...\n');
fprintf('预计生成矩阵大小: %d (Pixels) x %d (Inner) x %d (Angles).\n', length(valid_idx), Total_Inner, Nal);

tic;
% 预分配内存 (注意：如果内存不足，可改为 single 精度)
I_sim_all = zeros(length(valid_idx), Total_Inner, Nal); 

for n4 = 1:Nal
    % 取出当前角度的 Ms
    Ms_chunk = Ms1(:, :, :, :, n4);
    Ms_batch = reshape(Ms_chunk, 1936, Total_Inner);
    
    % 仿真: T x Ms
    I_sim_batch = T1_valid * Ms_batch;
    
    % 归一化 (按列/图像)
    max_vals = max(I_sim_batch, [], 1); 
    max_vals(max_vals == 0) = 1;
    I_sim_norm = I_sim_batch ./ max_vals;
    
    % 保存到三维矩阵
    I_sim_all(:,:,n4) = I_sim_norm;
    
    fprintf('强度计算进度: Angle n4 = %d / %d \n', n4, Nal);
end
toc;

%% 步骤 5: 保存数据
% 定义参数范围 (用于后续脚本显示)
R_Period = 855-30 + (0:20)*3; 
R_TCD    = 405-30 + (0:20)*3;
R_Height = 195-15 + (0:10)*3;
R_Angle  = 86-3 + (0:6);

fprintf('正在保存数据到 Intensity_Data.mat...\n');
save('Intensity_Data.mat', 'I_sim_all', 'I1_vec', ...
     'Ni', 'Nj', 'Nk', 'Nal', 'Total_Inner', ...
     'R_Period', 'R_TCD', 'R_Height', 'R_Angle', '-v7.3'); % -v7.3 有助于处理大文件
fprintf('保存完成！\n');