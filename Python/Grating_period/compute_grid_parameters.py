from d2M_function import RCWAForward, TinyDNet, compute_Iout_vectorized
import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Model3 import IoutCNNMLP
import hdf5storage
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F
import time


def compute_grid_parameters(low, high, steps, thickness_label, duty_cycle_label, rcwa, S_Encoder, Spec, M_Decoder,
                            I_out_real, device):
    lambda_cache = torch.linspace(low, high, steps=steps, device=device, dtype=rcwa.rdtype)
    thickness_label, duty_cycle_label = thickness_label, duty_cycle_label
    loss_list = []
    for lm in lambda_cache:
        ms_pred = rcwa(thickness_label, lm, duty_cycle_label)  # [W,16] 或 [W,16,T]
        ms_pred = ms_pred.reshape(121, 4, 4)
        I_out_pred = compute_Iout_vectorized(ms_pred, S_Encoder, Spec, M_Decoder, eps=1e-12)
        loss = torch.mean(torch.abs(I_out_pred - I_out_real))
        loss_list.append(loss.item())

    best_idx = int(torch.tensor(loss_list).argmin())
    best_lambda = lambda_cache[best_idx]
    best_loss = loss_list[best_idx]

    # 2A) 直接 nn.Parameter 测试三参数梯度 注意 单位是um
    raw_La = nn.Parameter(torch.tensor([best_lambda], device=device, dtype=rcwa.rdtype))
    thickness_label = torch.tensor(thickness_label, device=device, dtype=rcwa.rdtype)
    duty_cycle_label = torch.tensor(duty_cycle_label, device=device, dtype=rcwa.rdtype)

    optimizer = torch.optim.LBFGS([raw_La],
                                  lr=1.0, history_size=20, max_iter=20,
                                  line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-9, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        ms_pred = rcwa(thickness_label, raw_La, duty_cycle_label)  # [W,16] 或 [W,16,T]
        ms_pred = ms_pred.reshape(121, 4, 4)
        I_out_pred = compute_Iout_vectorized(ms_pred, S_Encoder, Spec, M_Decoder, eps=1e-12)
        loss = F.mse_loss(I_out_pred, I_out_real)
        loss.backward()
        return loss

    training_loss = []
    for it in range(3):
        loss = optimizer.step(closure)
        training_loss.append(loss.item())
        # print(f"Lambda_label={Lambda_label:.5f}")
        # print(f"[{it+1:03d}] loss={loss.item():.6e}, Lambda_pred={raw_La.item():.5f}")
    return training_loss, raw_La.item()