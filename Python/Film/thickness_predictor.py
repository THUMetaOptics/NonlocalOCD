import numpy as np
import hdf5storage
import torch
import matplotlib.pyplot as plt
from my_dataset import MyDataset
from Model3 import IoutCNNMLP, ResidualBlock, ICmosToThetaResNet, icmos_to_ms_resnet, weights_init
from theta2Ms import M_pol
from d2M_function2 import MuellerMatrixFilmSingleLayer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import time
import copy


device = "cpu"


class ThicknessPredictor:

    def __init__(self, I_out_real, Spec, S_Encoder, M_Decoder, Mfilm, device=None):
        self.exp_d_list = [100, 900, 200, 800, 300, 700, 400, 600, 500]
        self.I_out_real = I_out_real
        self.Spec = Spec
        self.S_Encoder = S_Encoder
        self.M_Decoder = M_Decoder
        self.Mfilm = Mfilm
        self.device = device or (I_out_real.device if hasattr(I_out_real, "device") else torch.device("cpu"))

        # 初始化模型
        layers = [2, 2, 2, 2, 2]
        self.model = icmos_to_ms_resnet(layers, out_dim=1, device=self.device).to(self.device)
        self.model.apply(weights_init)

        # 统一把输入搬到设备上
        self.I_out_real = self.I_out_real.to(self.device)
        self.Spec = self.Spec.to(self.device)
        self.S_Encoder = self.S_Encoder.to(self.device)
        self.M_Decoder = self.M_Decoder.to(self.device)

    def compute_Iout_vectorized(self, d, Sencoder, Spec, Mdecoder, eps=1e-12):
        # 扩展 Ms 到 [N, 256, 256, 121, 4, 4]
        if not torch.is_tensor(d):
            d = torch.tensor([[float(d)]], device=self.device, dtype=torch.float32)
        else:
            d = d.to(self.device).float().view(1, 1)
        Ms = self.Mfilm(d, 45)  # torch.Size([N, 4, 4, 121])
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

    # -------- Phase 1 预训练 --------
    def phase1_pre_training(self, exp_d, epochs=3):
        self.model.train()
        d_label = torch.ones((1, 1), device=self.device, dtype=torch.float32) * float(exp_d)
        criterion = nn.MSELoss()
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, history_size=20, max_iter=20,
                                      line_search_fn='strong_wolfe',
                                      tolerance_grad=1e-9, tolerance_change=1e-9)
        I = self.I_out_real

        def closure():
            optimizer.zero_grad(set_to_none=True)
            d = self.model(I)  # 期望输出标量或 [1,1]
            loss = criterion(d, d_label)
            loss.backward()
            return loss

        training_loss = []
        for _ in range(epochs):
            loss = optimizer.step(closure)
            training_loss.append(loss.item())
        return self.model, training_loss

    # -------- Phase 2: 端到端物理一致性微调 --------
    def phase2_finetune(self, epochs=3):
        self.model.train()
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, history_size=20, max_iter=20,
                                      line_search_fn='strong_wolfe',
                                      tolerance_grad=1e-9, tolerance_change=1e-9)
        I = self.I_out_real

        def closure():
            optimizer.zero_grad(set_to_none=True)
            d = self.model(I)
            I_pred = self.compute_Iout_vectorized(d, self.S_Encoder, self.Spec, self.M_Decoder)
            loss = torch.abs(I_pred - I).sum() / (I_pred.sum().clamp_min(1e-12))
            loss.backward()
            return loss

        training_loss = []
        for _ in range(epochs):
            loss = optimizer.step(closure)
            training_loss.append(loss.item())
        return self.model, training_loss

    def training_procedure(self, THRESH=1e-6, phase1_epochs=3, phase2_epochs=3):
        # 2) 不达标 → 遍历 exp_d_list：phase1 → phase2
        pred_d_list = []
        best_residual = float('inf')
        best_index = -1
        model_list = []
        I_list = []
        for i in range(len(self.exp_d_list)):
            # 每次使用新的exp_d的时候 需要重新初始化模型
            layers = [2, 2, 2, 2, 2]
            self.model = icmos_to_ms_resnet(layers, out_dim=1, device=self.device).to(self.device)
            self.model.apply(weights_init)

            self.model, phase1_loss = self.phase1_pre_training(self.exp_d_list[i], phase1_epochs)
            self.model, phase2_loss = self.phase2_finetune(phase2_epochs)

            with torch.no_grad():
                d = self.model(self.I_out_real)
                I_pred = self.compute_Iout_vectorized(d, self.S_Encoder, self.Spec, self.M_Decoder)
                residual = (abs(self.I_out_real.detach().numpy().squeeze() - I_pred.detach().numpy().squeeze())).sum()
            pred_d_list.append(d)
            I_list.append(I_pred)
            model_list.append(copy.deepcopy(self.model))
            if residual.item() < best_residual:
                best_residual = residual.item()
                best_index = i

            # 达标就提前返回
            if abs(residual.item()) < THRESH:
                # print("Training finished by pre-training phase 1 and finetune phase 2!")
                # print(f"the predicted thickness is：{pred_d_list[best_index].item()}")
                return pred_d_list[best_index], model_list[best_index], I_list[best_index], phase1_loss, phase2_loss

        # 3) 全部跑完，返回最优
        # print("Training finished by pre-training phase 1 and finetune phase 2!")
        # print(f"the predicted thickness is：{pred_d_list[best_index].item()}")
        return pred_d_list[best_index], model_list[best_index], I_list[best_index], phase1_loss, phase2_loss


def compute_Iout_vectorized(d, Sencoder, Spec, Mdecoder, eps=1e-12):
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
    Ms = Mfilm(ideal_d,45)  # torch.Size([N, 4, 4, 121])
    Ms = Ms.permute(0, 3, 1, 2)  # torch.Size([1, 121, 4, 4])
    Ms = Ms.unsqueeze(1).unsqueeze(1) # torch.Size([N, 1, 1, 121, 4, 4])
    Ms = Ms.float().to(device)

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