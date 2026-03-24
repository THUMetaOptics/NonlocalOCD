%% 主流程 (Main Pipeline)
clc; clear; close all;

% ==================== 1. 基础配置与公用数据加载 ====================
base_path = 'E:\Library_based\';
% addpath(genpath(base_path)); 

% 加载系统矩阵与光源谱
load([base_path, 'S_Encoder.mat'], 'S'); %编码部分标定结果
load([base_path, 'M_Decoder.mat'], 'M'); %解码部分标定结果
spec = load([base_path, 'spec_grating_source.mat']).Spec1;%光源谱修正

% 生成公用的 T 矩阵
T1_reshaped = generate_T_matrix(S, M);
spec_7d = reshape(spec, [1, 1, 121, 1, 1, 1, 1]);

% 加载并处理实验图像，提取有效区域
[I1_vec, T1_valid] = prepare_exp_image(base_path, T1_reshaped);


% ==================== 2. 执行多精度参数搜索 ====================

% --- 2.1 精度 3 nm ---
% R_Period_3 = 855-30 + (0:20)*3; 
% R_TCD_3    = 405-30 + (0:20)*3;
% R_Height_3 = 195-15 + (0:10)*3;
% R_Angle_3  = 86-3 + (0:6);
fprintf('\n>>> 开始处理穆勒矩阵库...\n');
load([base_path, 'Ms_grating_tra_all2.mat'], 'Ms');
Ms1_3 = preprocess_Ms(Ms, spec_7d, 21, 21, 11, 7);
clear Ms; 
% --- 2.2 精度 0.5 nm ---
% R_Period_0p5 = 861 + (0:12)*0.5; 
% R_TCD_0p5    = 402 + (0:12)*0.5;
% R_Height_0p5 = 195 + (0:12)*0.5;
% R_Angle_0p5  = 83 + (0:4)*0.5; 
load([base_path, 'Ms_step_0.5.mat'], 'Ms');
Ms1_0p5 = preprocess_Ms(Ms, spec_7d, 13, 13, 13, 5);
clear Ms;
% --- 2.3 精度 0.1 nm ---
% R_Period = 861.5 + (0:10)*0.1; 
% R_TCD    = 403 + (0:10)*0.1;
% R_Height = 196.5 + (0:10)*0.1;
% R_Angle  = 83 + (0:10)*0.1; 
load([base_path, 'Ms_step_0.1.mat'], 'Ms');
Ms1_0p1 = preprocess_Ms(Ms, spec_7d, 11, 11, 11, 11);
clear Ms;

tic;
fprintf('\n>>> 开始处理 3 nm 精度...\n');
error_3 = calc_error_matrix(Ms1_3, T1_valid, I1_vec, 21, 21, 11, 7, '3nm');
fprintf('\n>>> 开始处理 0.5 nm 精度...\n');
error_0p5 = calc_error_matrix(Ms1_0p5, T1_valid, I1_vec, 13, 13, 13, 5, '0.5nm');
fprintf('\n>>> 开始处理 0.1 nm 精度...\n');
error_0p1 = calc_error_matrix(Ms1_0p1, T1_valid, I1_vec, 11, 11, 11, 11, '0.1nm');
toc;

% ==================== 3. 最终结果提取 (基于 0.1nm) ====================
R_Period = 861.5 + (0:10)*0.1; 
R_TCD    = 403 + (0:10)*0.1;
R_Height = 196.5 + (0:10)*0.1;
R_Angle  = 83 + (0:10)*0.1; 

fprintf('\n------------------------------------------------\n');
[~, min_idx] = min(error_0p1(:));
[n1_b, n2_b, n3_b, n4_b] = ind2sub(size(error_0p1), min_idx);

fprintf('物理参数估算:\n');
fprintf('  P: %.1f, T: %.1f, H: %.1f, Ang: %.1f\n', ...
    R_Period(n1_b), R_TCD(n2_b), R_Height(n3_b), R_Angle(n4_b));
fprintf('------------------------------------------------\n');



%% =========================================================================
%                               局部函数定义区
% ==========================================================================

% 1. 生成 T 矩阵的函数
function T1_reshaped = generate_T_matrix(S, M)
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
end

% 2. 实验图像预处理函数
function [I1_vec, T1_valid] = prepare_exp_image(base_path, T1_reshaped)
    fprintf('加载实验图像并提取掩膜数据...\n');
    I1 = imread([base_path, 'phi0.png']); 
    bk = imread([base_path, 'bk.png']);

    I1 = Mean_Intensity_2dim(I1, 20);
    bk = Mean_Intensity_2dim(bk, 20);
    I1 = double(I1 - bk);

    temp_img = ones(size(I1));
    masked_template = Select_circle(temp_img, 128, 128, 128); 
    valid_idx = find(masked_template ~= 0); 

    I1_vec = I1(valid_idx);
    max_I1 = max(I1_vec);
    if max_I1 == 0, max_I1 = 1; end
    I1_vec = I1_vec / max_I1; 

    T1_valid = T1_reshaped(valid_idx, :); 
end

% 3. 统一预处理 Ms 库的函数
function Ms1 = preprocess_Ms(Ms_raw, spec_7d, Ni, Nj, Nk, Nal)
    % 1. 应用光谱
    for i = 1:4 
        for j = 1:4
            Ms_raw(i,j,:,:,:,:,:) = Ms_raw(i,j,:,:,:,:,:) .* spec_7d;
        end
    end
    
    % 2. 重塑为一维向量矩阵
    Ms1 = zeros(1936, Ni, Nj, Nk, Nal);
    for n4 = 1:Nal
        for n3 = 1:Nk
            for n2 = 1:Nj
                for n1 = 1:Ni
                    temp_block = Ms_raw(:,:,:,n1,n2,n3,n4); 
                    Ms1(:, n1, n2, n3, n4) = reshape(permute(temp_block, [3, 2, 1]), [], 1);
                end
            end
        end
    end
end

% 4. 计算 MSE 误差矩阵的函数
function error_flat = calc_error_matrix(Ms1, T1_valid, I1_vec, Ni, Nj, Nk, Nal, label)
    error_flat = zeros(Ni, Nj, Nk, Nal);
    Total_Inner = Ni * Nj * Nk; 
    

    for n4 = 1:Nal
        Ms_batch = reshape(Ms1(:, :, :, :, n4), 1936, Total_Inner);
        I_sim_batch = T1_valid * Ms_batch;
        
        max_vals = max(I_sim_batch, [], 1); 
        max_vals(max_vals == 0) = 1;
        I_sim_norm = I_sim_batch ./ max_vals;
        
        diff_sq = (I_sim_norm - I1_vec).^2;
        batch_errors = mean(diff_sq, 1);    
        
        error_flat(:, :, :, n4) = reshape(batch_errors, Ni, Nj, Nk);
        fprintf('[%s] 计算进度: %d / %d \n', label, n4, Nal);
    end
end