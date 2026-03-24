%% Step 1: 基础设置与数据加载 (保持不变)
clc; clear; close all;

% 路径配置 (请根据实际情况修改)
base_path = 'F:\Seafile\Research\Nonlocal\New_Mueller\Experiment\20251222_2\Alluse\Library_based\';
addpath(genpath(base_path)); 

% 加载系统矩阵
load([base_path, 'S_Encoder.mat']);%编码部分空间偏振光谱调制
load([base_path, 'M_Decoder.mat']);%解码部分空间偏振光谱调制
spec = load([base_path, 'spec_grating_source.mat']).Spec1;%光源谱
% 重组 T 矩阵 (T = M .* S)
fprintf('正在初始化 T 矩阵...\n');
T = zeros(256, 256, 121, 4, 4);
for i = 1:121
    for j = 1:4
        for k =1:4
            T(:,:,i,j,k) = M(:,:,i,j) .* S(:,:,i,k);
        end
    end
end

% 重塑 T1 为 (Pixels x 1936)
T1 = zeros(256, 256, 1936);
for j = 1:4
    for k =1:4
        for i = 1 :121
            T1(:,:,(j-1)*4*121+(k-1)*121+i) = T(:,:,i,j,k);
        end
    end
end
T1_reshaped = reshape(T1, [], 1936); 

fprintf('正在预处理 Ms 库...\n');
load([base_path, 'Ms_grating_tra_all2.mat']); % 加载新的 Ms (7维数据)
Ms_step_3 = Ms;

load([base_path, 'Ms_step_0.5.mat']); % 加载新的 Ms (7维数据)
Ms_step_0p5 = Ms;
% fprintf('正在预处理 Ms 库...\n');

load([base_path, 'Ms_step_0.1.mat']); % 加载新的 Ms (7维数据)
Ms_step_0p1 = Ms;

% fprintf('正在预处理 Ms 库...\n');
%% Step 2: Mueller 矩阵库预处理 (Ms -> Ms1) [已针对 7维数据 修改]



range_i = 1:21;   % n1: Period
range_j = 1:21;   % n2: TCD
range_k = 1:11;   % n3: Height
range_al = 1:7;   % n4: Angle (原Alpha，现在代表唯一的角度参数)
% range_be 已移除

Ni = length(range_i); Nj = length(range_j); Nk = length(range_k);
Nal = length(range_al); 

% 应用光谱 (Spec) - 维度从 8 变 7
for i = 1:4 
    for j = 1:4
        % 修正 reshape 维度以匹配新的 Ms (4,4,121, Ni,Nj,Nk,Nal)
        Ms_step_3(i,j,:,:,:,:,:) = Ms_step_3(i,j,:,:,:,:,:) .* reshape(spec, [1, 1, 121, 1, 1, 1, 1]);
    end
end

% 重塑 Ms1 为 (1936 x Ni x Nj x Nk x Nal) - 移除了 Nbe 维度
Ms1 = zeros(1936, Ni, Nj, Nk, Nal);

% 循环层数减少一层
for n4 = 1:Nal
    for n3 = 1:Nk
        for n2 = 1:Nj
            for n1 = 1:Ni
                % 提取 4x4x121 并拉直
                % 注意这里 Ms 索引减少了一个维度
                temp_block = Ms_step_3(:,:,:,n1,n2,n3,n4); 
                
                % Permute 顺序保持不变 (i, k, j) 以匹配 T1
                temp_permuted = permute(temp_block, [3, 2, 1]); 
                Ms1(:, n1, n2, n3, n4) = temp_permuted(:);
            end
        end
    end
end

%% Step 3: 实验图像加载与 4维误差计算 (核心计算层) [已修改为 MSE]
fprintf('加载实验图像并计算误差矩阵...\n');

% 3.1 加载图像
I1 = imread([base_path, 'phi0.png']); 
bk = imread([base_path, 'bk.png']);

I1 = Mean_Intensity_2dim(I1, 20);
bk = Mean_Intensity_2dim(bk, 20);
I1 = double(I1 - bk);

% 3.2 生成 Mask (圆)
temp_img = ones(size(I1));
masked_template = Select_circle(temp_img, 128, 128, 128); 
valid_idx = find(masked_template ~= 0); 

% 3.3 提取有效数据向量
I1_vec = I1(valid_idx);
max_I1 = max(I1_vec);
if max_I1 == 0, max_I1 = 1; end
I1_vec = I1_vec / max_I1; % 归一化

T1_valid = T1_reshaped(valid_idx, :); 

% 3.4 向量化误差计算 - 维度变为 4维 (Ni, Nj, Nk, Nal)
error_flat = zeros(Ni, Nj, Nk, Nal);
Total_Inner = Ni * Nj * Nk; 
fprintf('参数搜索精度3 nm...\n');
tic;
% 只需要循环 n4 (Angle)
for n4 = 1:Nal
    % 取出切片 (1936 x Inner_Batch)
    Ms_chunk = Ms1(:, :, :, :, n4);
    Ms_batch = reshape(Ms_chunk, 1936, Total_Inner);
    
    % 仿真
    I_sim_batch = T1_valid * Ms_batch;
    
    % 归一化 (按列)
    max_vals = max(I_sim_batch, [], 1); 
    max_vals(max_vals == 0) = 1;
    I_sim_norm = I_sim_batch ./ max_vals;
    
    diff_sq = (I_sim_norm - I1_vec).^2; % 平方差
    batch_errors = mean(diff_sq, 1);    % 求均值 (MSE)
    % --------------------------------
    
    % 存入
    error_flat(:, :, :, n4) = reshape(batch_errors, Ni, Nj, Nk);
    
    fprintf('计算进度: n4 = %d / %d \n', n4, Nal);
end
toc;


%% Step 4: 结果选择器 (Result Selector) [参数范围已更新]
% --- 【更新的部分】参数范围定义 ---
R_Period = 855-30 + (0:20)*3; 
R_TCD    = 405-30 + (0:20)*3;
R_Height = 195-15 + (0:10)*3;
R_Angle  = 86-3 + (0:6); % 原 Alpha/Beta 统一为一个角度参数

%%



range_i = 1:13;   % n1: Period
range_j = 1:13;   % n2: TCD
range_k = 1:13;   % n3: Height
range_al = 1:5;   % n4: Angle
% range_be 已移除

Ni = length(range_i); Nj = length(range_j); Nk = length(range_k);
Nal = length(range_al); 

% 应用光谱 (Spec) - 维度从 8 变 7
for i = 1:4 
    for j = 1:4
        % 修正 reshape 维度以匹配新的 Ms (4,4,121, Ni,Nj,Nk,Nal)
        Ms_step_0p5(i,j,:,:,:,:,:) = Ms_step_0p5(i,j,:,:,:,:,:) .* reshape(spec, [1, 1, 121, 1, 1, 1, 1]);
    end
end

% 重塑 Ms1 为 (1936 x Ni x Nj x Nk x Nal) - 移除了 Nbe 维度
Ms1 = zeros(1936, Ni, Nj, Nk, Nal);

% 循环层数减少一层
for n4 = 1:Nal
    for n3 = 1:Nk
        for n2 = 1:Nj
            for n1 = 1:Ni
                % 提取 4x4x121 并拉直
                % 注意这里 Ms 索引减少了一个维度
                temp_block = Ms_step_0p5(:,:,:,n1,n2,n3,n4); 
                
                % Permute 顺序保持不变 (i, k, j) 以匹配 T1
                temp_permuted = permute(temp_block, [3, 2, 1]); 
                Ms1(:, n1, n2, n3, n4) = temp_permuted(:);
            end
        end
    end
end
%%
error_flat = zeros(Ni, Nj, Nk, Nal);
Total_Inner = Ni * Nj * Nk; 
fprintf('参数搜索精度0.5 nm...\n');
tic;
% 只需要循环 n4 (Angle)
for n4 = 1:Nal
    % 取出切片 (1936 x Inner_Batch)
    Ms_chunk = Ms1(:, :, :, :, n4);
    Ms_batch = reshape(Ms_chunk, 1936, Total_Inner);
    
    % 仿真
    I_sim_batch = T1_valid * Ms_batch;
    
    % 归一化 (按列)
    max_vals = max(I_sim_batch, [], 1); 
    max_vals(max_vals == 0) = 1;
    I_sim_norm = I_sim_batch ./ max_vals;
    
    diff_sq = (I_sim_norm - I1_vec).^2; % 平方差
    batch_errors = mean(diff_sq, 1);    % 求均值 (MSE)
    % --------------------------------
    
    % 存入
    error_flat(:, :, :, n4) = reshape(batch_errors, Ni, Nj, Nk);
    
    fprintf('计算进度: n4 = %d / %d \n', n4, Nal);
end
toc;
%%
R_Period = 861 + (0:12)*0.5; 
R_TCD    = 402 + (0:12)*0.5;
R_Height = 195 + (0:12)*0.5;
R_Angle  = 83 + (0:4)*0.5; 

%%


range_i = 1:11;   % n1: Period
range_j = 1:11;   % n2: TCD
range_k = 1:11;   % n3: Height
range_al = 1:11;   % n4: Angle (原Alpha，现在代表唯一的角度参数)

Ni = length(range_i); Nj = length(range_j); Nk = length(range_k);
Nal = length(range_al); 

% 应用光谱 (Spec) - 维度从 8 变 7
for i = 1:4 
    for j = 1:4
        % 修正 reshape 维度以匹配新的 Ms (4,4,121, Ni,Nj,Nk,Nal)
        Ms_step_0p1(i,j,:,:,:,:,:) = Ms_step_0p1(i,j,:,:,:,:,:) .* reshape(spec, [1, 1, 121, 1, 1, 1, 1]);
    end
end

% 重塑 Ms1 为 (1936 x Ni x Nj x Nk x Nal) - 移除了 Nbe 维度
Ms1 = zeros(1936, Ni, Nj, Nk, Nal);

% 循环层数减少一层
for n4 = 1:Nal
    for n3 = 1:Nk
        for n2 = 1:Nj
            for n1 = 1:Ni
                % 提取 4x4x121 并拉直
                % 注意这里 Ms 索引减少了一个维度
                temp_block = Ms_step_0p1(:,:,:,n1,n2,n3,n4); 
                
                % Permute 顺序保持不变 (i, k, j) 以匹配 T1
                temp_permuted = permute(temp_block, [3, 2, 1]); 
                Ms1(:, n1, n2, n3, n4) = temp_permuted(:);
            end
        end
    end
end
%%
error_flat = zeros(Ni, Nj, Nk, Nal);
Total_Inner = Ni * Nj * Nk; 
fprintf('参数搜索精度0.1 nm...\n');
tic;
% 只需要循环 n4 (Angle)
for n4 = 1:Nal
    % 取出切片 (1936 x Inner_Batch)
    Ms_chunk = Ms1(:, :, :, :, n4);
    Ms_batch = reshape(Ms_chunk, 1936, Total_Inner);
    
    % 仿真
    I_sim_batch = T1_valid * Ms_batch;
    
    % 归一化 (按列)
    max_vals = max(I_sim_batch, [], 1); 
    max_vals(max_vals == 0) = 1;
    I_sim_norm = I_sim_batch ./ max_vals;
    
    
    diff_sq = (I_sim_norm - I1_vec).^2; % 平方差
    batch_errors = mean(diff_sq, 1);    % 求均值 (MSE)
    % --------------------------------
    
    % 存入
    error_flat(:, :, :, n4) = reshape(batch_errors, Ni, Nj, Nk);
    
    fprintf('计算进度: n4 = %d / %d \n', n4, Nal);
end
toc;
%%
R_Period = 861.5 + (0:10)*0.1; 
R_TCD    = 403 + (0:10)*0.1;
R_Height = 196.5 + (0:10)*0.1;
R_Angle  = 83 + (0:10)*0.1; % 原 Alpha/Beta 统一为一个角度参数
fprintf('\n------------------------------------------------\n');
[min_val_curr, min_idx] = min(error_flat(:));
[n1_b, n2_b, n3_b, n4_b] = ind2sub(size(error_flat), min_idx);


% 统一输出：Best Indices (4个)
best_indices = [n1_b, n2_b, n3_b, n4_b];

fprintf('选定的最佳参数索引: [%d, %d, %d, %d]\n', best_indices);
fprintf('物理参数估算:\n');
fprintf('  P: %.1f, T: %.1f, H: %.1f, Ang: %.1f\n', ...
    R_Period(n1_b), R_TCD(n2_b), R_Height(n3_b), R_Angle(n4_b));
fprintf('------------------------------------------------\n');
