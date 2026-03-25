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
from theta2Ms import M_pol, R, M_wp
import torch
import torch.nn as nn
import torch.optim as optim


class PINN():

    def __init__(self, Mfilm_path, Sin_path, Spec_path, I_out_path):

        self.device = "cpu"
        M_dict = hdf5storage.loadmat(Mfilm_path)
        Mfilm = M_dict["M"]
        Mfilm = torch.from_numpy(Mfilm)

        Mfilm = Mfilm.permute(0, 1, 3, 2)
        self.Mfilm = Mfilm.reshape(256, 256, 4 * 121)
        self.Mfilm = self.Mfilm.float().to(self.device)

        Sin_dict = hdf5storage.loadmat(Sin_path)
        Sin = Sin_dict["S"]
        Sin = torch.from_numpy(Sin)
        Sin = Sin.permute(0, 1, 2, 3)
        self.Sin = Sin.unsqueeze(4).unsqueeze(0)
        self.Sin = self.Sin.float().to(self.device)

        Spec_dict = hdf5storage.loadmat(Spec_path)
        Spec = Spec_dict["spec"]
        Spec = np.array(Spec, dtype=np.float64)
        Spec = torch.from_numpy(Spec)
        self.Spec = Spec.float().to(self.device)

        I_dict = hdf5storage.loadmat(I_out_path)
        Iout_0 = I_dict["I1"]
        Iout_0 = torch.from_numpy(Iout_0)
        I_out_real = Iout_0.to(self.device).float().unsqueeze(0)
        I_out_real[:, :, 251:256] = 0
        I_out_real[:, 87:100, 62:78] = 0
        self.I_out_real = I_out_real

        """加载模型"""
        layers = [2, 2, 2, 2, 2]
        # 创建模型实例
        self.model = icmos_to_ms_resnet(layers)

    def compute_Iout_vectorized(self, Ms, Sin, Spec, Mfilm):
        """
        计算 I_out 从 Ms, Sin, Spec, 和 Mfilm，使用向量化操作。
        # 241 —— 121
        参数:
            Ms (torch.Tensor): 形状为 [N, 1, 1, 241, 4, 4] 的张量。
            Sin (torch.Tensor): 形状为 [1, 256, 256, 241, 4, 1] 的张量。
            Spec (torch.Tensor): 形状为 [256, 256, 964] 的张量。
            Mfilm (torch.Tensor): 形状为 [256, 256, 964] 的张量。

        返回:
            I_out (torch.Tensor): 形状为 [N, 256, 256] 的张量。
        """
        N = Ms.shape[0]
        # 扩展 Ms 到 [N, 256, 256, 121, 4, 4]
        Ms_expanded = Ms.expand(N, 256, 256, 121, -1, -1).float()  # [N, 256, 256, 121, 4, 4]
        Sin = Sin.float()
        # 执行矩阵乘法，得到 [N, 256, 256, 121, 4, 1]
        S_out = torch.matmul(Ms_expanded, Sin)  # [N, 256, 256, 121, 4, 1]
        # 去除最后一个维度，得到 [N, 256, 256, 121, 4]
        S_out = S_out.squeeze(-1)  # [N, 256, 256, 121, 4]
        # 调整维度顺序，得到 [N, 256, 256, 4, 241]
        S_out = S_out.permute(0, 1, 2, 4, 3)  # [N, 256, 256, 4, 121]
        # 重塑为 [N, 256, 256, 964]
        S_out = S_out.reshape(N, 256, 256, 484)  # [N, 256, 256, 964]
        # 元素级相乘
        S_out_spec = S_out * Spec  # [N, 256, 256, 484]
        # 元素级相乘并在最后一个维度求和，得到 [N, 256, 256]
        I_out = (Mfilm * S_out_spec).sum(dim=-1)  # [N, 256, 256]
        I_out[:, :, 251:256] = 0
        I_out[:, 87:100, 62:78] = 0
        # I(87:100,62:78,:)=0;
        # # 归一化，每个样本独立归一化
        I_out = I_out / I_out.view(N, -1).max(dim=1, keepdim=True)[0].view(N, 1, 1)  # [N, 256, 256]
        return I_out

    def traversal_alrogithm(self, I_out_real):
        criterion = nn.MSELoss()
        loss_cache = float('inf')
        best_theta = float('inf')
        loss_list = []
        for theta in range(0, 181, 1):
            theta_degree = torch.ones([1, 1]) * theta
            Ms = M_pol(theta_degree)
            Ms = Ms.unsqueeze(1).unsqueeze(1)
            I_out_pred = self.compute_Iout_vectorized(Ms, self.Sin, self.Spec, self.Mfilm)  # [N, 256, 256]
            loss = criterion(I_out_pred, I_out_real)
            loss_list.append(loss.item())
            if loss.item() < loss_cache:
                loss_cache = loss.item()
                best_theta = theta_degree

        return best_theta

    def training(self, pre_train_lr, optimization_lr):

        """遍历搜索阶段"""
        best_theta = self.traversal_alrogithm(self.I_out_real)
        print(f"step1: {best_theta}")

        """预训练阶段"""
        criterion = nn.MSELoss()
        initial_lr = pre_train_lr  # 提高初始学习率，增加探索性
        optimizer = optim.Adam(self.model.parameters(), lr=initial_lr)
        num_epochs = 300
        training_loss = []
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            optimizer.zero_grad()
            theta_pred = self.model(self.I_out_real)
            loss = criterion(theta_pred, best_theta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 控制梯度大小，防止梯度爆炸
            optimizer.step()
            running_loss += loss.item()
            optimizer.zero_grad()
            epoch_loss = running_loss / (epoch + 1)
            training_loss.append(epoch_loss)

        plt.figure()
        plt.plot(training_loss)
        plt.yscale("log")
        plt.show()
        theta_pred = self.model(self.I_out_real)
        print(f"step2: {theta_pred}")

        """优化训练阶段"""
        num_epochs = 300
        training_loss = []
        criterion = nn.MSELoss()
        initial_lr = optimization_lr
        optimizer = optim.Adam(self.model.parameters(), lr=initial_lr)
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            optimizer.zero_grad()
            theta = self.model(self.I_out_real)
            Ms = M_pol(theta)
            Ms = Ms.unsqueeze(1).unsqueeze(1)
            I_out_pred = self.compute_Iout_vectorized(Ms, self.Sin, self.Spec, self.Mfilm)  # [N, 256, 256]
            loss = criterion(I_out_pred, self.I_out_real)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 控制梯度大小，防止梯度爆炸
            optimizer.step()
            running_loss += loss.item()
            optimizer.zero_grad()

            epoch_loss = running_loss / (epoch + 1)
            training_loss.append(epoch_loss)

        theta = self.model(self.I_out_real)
        print("Step3 predicted theta is {:.4f}".format(theta[0][0].item()))
        plt.figure()
        plt.plot(training_loss)
        plt.yscale("log")
        plt.show()

        return theta[0][0].item()

