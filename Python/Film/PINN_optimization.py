#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: PopSama
@Date: 2025/1/13
@Description: 
"""
import numpy as np
import hdf5storage
import torch
import matplotlib.pyplot as plt
from my_dataset import MyDataset
from Model3 import ResidualBlock, ICmosToThetaResNet, icmos_to_ms_resnet, weights_init
from theta2Ms import M_pol
from d2M_function2 import MuellerMatrixFilmSingleLayer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


class PINN():

    def __init__(self, Mfilm_path, Sin_path, Spec_path, I_out_path):

        self.device = "cpu"
        M_dict = hdf5storage.loadmat(Mfilm_path)
        M_Decoder = M_dict["M"]
        M_Decoder = torch.from_numpy(M_Decoder)
        M_Decoder = M_Decoder.permute(0, 1, 3, 2)
        M_Decoder = M_Decoder.reshape(256, 256, 4 * 121)
        self.M_Decoder = M_Decoder
        self.M_Decoder = self.M_Decoder.float().to(self.device)

        Sencoder_dict = hdf5storage.loadmat(Sin_path)
        S_Encoder = Sencoder_dict["S"]
        S_Encoder = torch.from_numpy(S_Encoder)
        S_Encoder = S_Encoder.unsqueeze(4).unsqueeze(0)
        self.S_Encoder = S_Encoder
        self.S_Encoder = self.S_Encoder.float().to(self.device)

        Spec_dict = hdf5storage.loadmat(Spec_path)
        Spec = Spec_dict["spec"]
        Spec = np.array(Spec, dtype=np.float64)
        Spec = torch.from_numpy(Spec)
        self.Spec = Spec.float().to(self.device)

        I_dict = hdf5storage.loadmat(I_out_path)
        Iout_0 = I_dict["I1"]
        Iout_0 = torch.from_numpy(Iout_0)
        I_out_real = Iout_0.to(self.device).float().unsqueeze(0)
        self.I_out_real = I_out_real

        """加载模型"""
        layers = [2, 2, 2, 2, 2]
        self.model = icmos_to_ms_resnet(layers, out_dim=1)
        self.model.apply(weights_init)

        # Ms function
        self.Mfilm = MuellerMatrixFilmSingleLayer(
            si_re_path=r"F:\科研项目\XueXinyuan\D2M_plus\Training\Training_Mfilm\dataset\Si_re.txt",
            si_im_path=r"F:\科研项目\XueXinyuan\D2M_plus\Training\Training_Mfilm\dataset\Si_Im.txt",
            device="cuda"
        )

    def compute_Iout_vectorized(self, d, Sencoder, Spec, Mdecoder, eps=1e-12):
        """
        计算 I_out 从 Ms, Sencoder, Spec, 和 Mdecoder，使用向量化操作。
        参数:
            Ms (torch.Tensor): 形状为 [N, 1, 1, 121, 4, 4] 的张量。
            Sencoder (torch.Tensor): 形状为 [1, 256, 256, 121, 4, 1] 的张量。
            Spec (torch.Tensor): 形状为 [1, 484]] 的张量。
            Mdecoder (torch.Tensor): 形状为 [256, 256, 484] 的张量。

        返回:
            I_out (torch.Tensor): 形状为 [N, 256, 256] 的张量。
        """

        # 扩展 Ms 到 [N, 256, 256, 121, 4, 4]

        ideal_d = torch.ones(1, 1) * d
        Ms = self.Mfilm(ideal_d, 45)  # torch.Size([N, 4, 4, 121])
        Ms = Ms.permute(0, 3, 1, 2)  # torch.Size([1, 121, 4, 4])
        Ms = Ms.unsqueeze(1).unsqueeze(1)  # torch.Size([N, 1, 1, 121, 4, 4])
        Ms = Ms.float().to(self.device)

        N = Ms.shape[0]
        # 扩展 Ms 到 [N, 256, 256, 121, 4, 4]

        Ms_expanded = Ms.expand(N, 256, 256, -1, -1, -1).float()  # [N, 256, 256, 121, 4, 4]

        Sencoder = Sencoder.float()

        # 执行矩阵乘法，得到 [N, 256, 256, 121, 4, 1]
        S_out = torch.matmul(Ms_expanded, Sencoder)  # [N, 256, 256, 121, 4, 1]

        # # 去除最后一个维度，得到 [N, 256, 256, 121, 4]
        S_out = S_out.squeeze(-1)  # [N, 256, 256, 121, 4]

        # # 调整维度顺序，得到 [N, 256, 256, 4, 241]
        S_out = S_out.permute(0, 1, 2, 4, 3)  # [N, 256, 256, 4, 121]

        # # 重塑为 [N, 256, 256, 484]
        S_out = S_out.reshape(N, 256, 256, -1)  # [N, 256, 256, 484]

        # # 元素级相乘
        Spec = Spec.squeeze()
        # print(Spec.shape)
        S_out_spec = S_out * Spec  # [N, 256, 256, 484]
        # print(S_out_spec.shape)

        # # # 元素级相乘并在最后一个维度求和，得到 [N, 256, 256]
        I_out = (Mdecoder * S_out_spec).sum(dim=-1)  # [N, 256, 256]

        # # # # 归一化，每个样本独立归一化
        I_out = I_out / I_out.view(N, -1).max(dim=1, keepdim=True)[0].view(N, 1, 1)  # [N, 256, 256]
        return I_out

    def training(self, expected_d, pre_train_lr, optimization_lr):

        """预训练阶段"""
        d_label = torch.ones([1, 1]) * expected_d
        # 训练参数
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=pre_train_lr)
        num_epochs = 200
        training_loss = []

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            # 1) 预测 d
            d = self.model(self.I_out_real)
            # 3) 计算总损失
            loss = criterion(d, d_label)
            loss.backward()
            running_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 控制梯度大小，防止梯度爆炸
            optimizer.step()
            epoch_loss = running_loss / (epoch + 1)
            training_loss.append(epoch_loss)
        plt.figure()
        plt.plot(training_loss)
        plt.yscale("log")
        plt.show()
        d = self.model(self.I_out_real)
        print("step1: {} ".format(d.item()))


        """优化训练阶段"""
        best_loss = float('inf')  # 初始化最佳损失为正无穷大
        best_epoch = -1  # 初始化最佳 epoch
        optimizer = optim.Adam(self.model.parameters(), lr=optimization_lr)
        # 训练循环
        num_epochs = 60
        training_loss = []
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            # 1) 预测 d
            d = self.model(self.I_out_real)
            # 2) 物理层前向
            Spec = self.Spec.float().to(self.device)
            I_out_pred = self.compute_Iout_vectorized(d, self.S_Encoder, Spec, self.M_Decoder, eps=1e-12)  # [N, 256, 256]
            # 3) 计算总损失
            loss = torch.abs(I_out_pred - self.I_out_real).sum()
            loss.backward()
            training_loss.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 控制梯度大小，防止梯度爆炸
            optimizer.step()
        plt.figure()
        plt.plot(training_loss)
        plt.yscale("log")
        plt.show()

        return d, self.model


#!/usr/bin/env python
# -*- coding: utf-8 -*-


torch.backends.cudnn.benchmark = True  # 提升卷积/归一化等性能

class PINN_GPU:
    def __init__(self, Mfilm_path, Sin_path, Spec_path, I_out_path):
        # —— 强制 CUDA —— #
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用：请在有 GPU 的环境中运行 PINN_GPU。")
        self.device = torch.device("cuda")
        self.dtype  = torch.float32

        # Mdecoder: [256,256,121,4] -> [256,256,4,121] -> [256,256,484]（j-major）
        M_dict = hdf5storage.loadmat(Mfilm_path)
        M_Decoder = torch.from_numpy(M_dict["M"]).to(self.dtype)
        M_Decoder = M_Decoder.permute(0, 1, 3, 2).contiguous()        # [256,256,4,121]
        M_Decoder = M_Decoder.view(256, 256, 4*121)                    # [256,256,484]
        self.M_Decoder = M_Decoder.to(self.device)

        # Sencoder: [256,256,121,4] -> [1,256,256,121,4,1]
        Sencoder_dict = hdf5storage.loadmat(Sin_path)
        S_Encoder = torch.from_numpy(Sencoder_dict["S"]).to(self.dtype)  # [256,256,121,4]
        S_Encoder = S_Encoder.unsqueeze(4).unsqueeze(0).contiguous()     # [1,256,256,121,4,1]
        self.S_Encoder = S_Encoder.to(self.device)

        # Spec: 预期 [1,484] 或 [484]
        Spec_dict = hdf5storage.loadmat(Spec_path)
        Spec = np.array(Spec_dict["spec"], dtype=np.float32)
        Spec = torch.from_numpy(Spec).to(self.dtype)
        self.Spec = Spec.to(self.device)

        # I_out_real: [256,256] -> [1,256,256]
        I_dict = hdf5storage.loadmat(I_out_path)
        Iout_0 = torch.from_numpy(I_dict["I1"]).to(self.dtype)     # [256,256]
        self.I_out_real = Iout_0.unsqueeze(0).to(self.device)      # [1,256,256]

        # 模型
        layers = [2, 2, 2, 2, 2]
        self.model = icmos_to_ms_resnet(layers, out_dim=1).to(self.device)
        self.model.apply(weights_init)

        # 物理 Ms 函数（确保与 self.device 一致）
        self.Mfilm = MuellerMatrixFilmSingleLayer(
            si_re_path=r"F:\科研项目\XueXinyuan\D2M_plus\Training\Training_Mfilm\dataset\Si_re.txt",
            si_im_path=r"F:\科研项目\XueXinyuan\D2M_plus\Training\Training_Mfilm\dataset\Si_Im.txt",
            device="cuda"  # 内部实现如需自行管理 device，保持为 cuda；输出会放到 GPU
        )

    @torch.no_grad()
    def _print_device_dtype(self):
        print(f"device={self.device}, dtype={self.dtype}")
        print(f"M_Decoder: {self.M_Decoder.device}, {self.M_Decoder.dtype}, {tuple(self.M_Decoder.shape)}")
        print(f"S_Encoder: {self.S_Encoder.device}, {self.S_Encoder.dtype}, {tuple(self.S_Encoder.shape)}")
        print(f"Spec     : {self.Spec.device}, {self.Spec.dtype}, {tuple(self.Spec.shape)}")
        print(f"I_real   : {self.I_out_real.device}, {self.I_out_real.dtype}, {tuple(self.I_out_real.shape)}")

    def compute_Iout_vectorized(self, d, Sencoder, Spec, Mdecoder, eps=1e-12):
        """
        d: [1,1] 或标量张量（GPU, float32）
        Sencoder: [1,256,256,121,4,1] (GPU,float32)
        Spec: [1,484] / [484]（GPU,float32）
        Mdecoder: [256,256,484]（GPU,float32）
        return: I_out [N,256,256]（N=1）
        """
        # —— Ms(d) —— #
        # 保持梯度链条：d 需要 requires_grad=True
        ones_11 = torch.ones((1, 1), device=self.device, dtype=self.dtype)
        ideal_d = ones_11 * (d.to(self.device, self.dtype) if torch.is_tensor(d) else float(d))
        Ms = self.Mfilm(ideal_d, 45)                         # 预期 [N,4,4,121] on CUDA
        Ms = Ms.permute(0, 3, 1, 2).contiguous()             # [N,121,4,4]
        Ms = Ms.unsqueeze(1).unsqueeze(1)                    # [N,1,1,121,4,4]
        Ms = Ms.to(self.device, self.dtype)
        N = Ms.shape[0]

        # —— 与 S 做 matmul —— #
        Ms_expanded = Ms.expand(N, 256, 256, -1, -1, -1)     # [N,256,256,121,4,4]
        Sencoder = Sencoder.to(self.device, self.dtype)
        S_out = torch.matmul(Ms_expanded, Sencoder)          # [N,256,256,121,4,1]
        S_out = S_out.squeeze(-1)                            # [N,256,256,121,4]
        S_out = S_out.permute(0, 1, 2, 4, 3).contiguous()    # [N,256,256,4,121]
        S_out = S_out.view(N, 256, 256, 4*121)               # [N,256,256,484]

        # —— 乘 spec —— #
        spec_flat = Spec.reshape(-1).to(self.device, self.dtype)      # [484]
        S_out_spec = S_out * spec_flat.view(1, 1, 1, 484)             # [N,256,256,484]

        # —— 与 Mdecoder 点乘并汇总 —— #
        Mdecoder = Mdecoder.to(self.device, self.dtype)                # [256,256,484]
        I_out = (Mdecoder * S_out_spec).sum(dim=-1)                    # [N,256,256]

        # —— 每样本最大值归一化（与你的数据保持一致） —— #
        denom = I_out.view(N, -1).max(dim=1, keepdim=True)[0].clamp_min(eps)
        I_out = I_out / denom.view(N, 1, 1)
        return I_out

    def training(self, expected_d, pre_train_lr, optimization_lr):
        # ========== 预训练阶段：MSE(d, d_label) ==========
        d_label = torch.full((1, 1), float(expected_d), device=self.device, dtype=self.dtype)  # [1,1]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=pre_train_lr)

        num_epochs = 200
        training_loss = []

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            d = self.model(self.I_out_real)      # [1,1], sigmoid 输出 0~1

            loss = criterion(d, d_label)         # 标量
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            training_loss.append(loss.item())

        plt.figure()
        plt.plot(training_loss)
        plt.yscale("log")
        plt.title("Pretrain: MSE(d, label)")
        plt.show()

        with torch.no_grad():
            d_now = self.model(self.I_out_real).item()
            print(f"step1 (pretrain) d≈ {d_now:.3f} nm")

        # ========== 优化训练阶段：L1(I_pred, I_real) ==========
        optimizer = optim.Adam(self.model.parameters(), lr=optimization_lr)
        num_epochs = 60
        training_loss = []

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            # 1) 预测 d
            d = self.model(self.I_out_real)      # [1,1]

            # 2) 物理前向
            I_out_pred = self.compute_Iout_vectorized(d, self.S_Encoder, self.Spec, self.M_Decoder)  # [1,256,256]

            # 3) L1 损失（标量）
            loss = torch.mean(torch.abs(I_out_pred - self.I_out_real))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            training_loss.append(loss.item())

        plt.figure()
        plt.plot(training_loss)
        plt.yscale("log")
        plt.title("Optimization: L1(I_pred, I_real)")
        plt.show()

        return d, self.model


if __name__ == "__main__":
    layers = [2, 2, 2, 2, 2]
    model = icmos_to_ms_resnet(layers, out_dim=1)
    model.apply(weights_init)

    x = torch.randn(10, 256, 256)
    d = model(x)
    print("输出尺寸:", d.shape)              # torch.Size([10, 1])
    print("d范围(≈0~1000):", d.min().item(), d.max().item())