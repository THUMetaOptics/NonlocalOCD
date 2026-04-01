# -*- coding: utf-8 -*-
import os, json, time
import torch
import numpy as np
import hdf5storage
from torch.quasirandom import SobolEngine
from d2M_function import RCWAForward
import math
from tqdm.auto import tqdm


def compute_Iout_vectorized(ms, Sencoder, Spec, Mdecoder, *, eps=1e-12):
    if not isinstance(ms, torch.Tensor):
        raise TypeError("ms must be a torch.Tensor")
    ref_dev, ref_dtype = ms.device, ms.dtype
    Sencoder = torch.as_tensor(Sencoder, device=ref_dev, dtype=ref_dtype)
    Spec     = torch.as_tensor(Spec,     device=ref_dev, dtype=ref_dtype)
    Mdecoder = torch.as_tensor(Mdecoder, device=ref_dev, dtype=ref_dtype)

    # ms = N, 121, 16
    ms = ms.reshape(-1, 121, 4, 4)
    ms = ms.unsqueeze(1).unsqueeze(1)                 # [N,1,  1,  121,4,4]
    N = ms.shape[0]
    Ms_expanded = ms.expand(N, 256, 256, -1, -1, -1)  # [N,256,256,121,4,4]
    S_out = torch.matmul(Ms_expanded, Sencoder)       # [N,256,256,121,4,1]
    S_out = S_out.squeeze(-1).permute(0, 1, 2, 4, 3)  # [N,256,256,4,121]
    S_out = S_out.reshape(N, 256, 256, -1)            # [N,256,256,484]
    S_out_spec = S_out * Spec.squeeze()               # [N,256,256,484]
    I_out = (Mdecoder * S_out_spec).sum(dim=-1)       # [N,256,256]
    eps_t = torch.as_tensor(eps, device=ref_dev, dtype=ref_dtype)
    denom = I_out.view(N, -1).max(dim=1, keepdim=True)[0].view(N, 1, 1).clamp_min(eps_t)
    I_out = I_out / denom
    return I_out


DEVICE = "cuda"
# ===== 修改这 3 个路径 =====
M_path = r"/WORK/sunliq_work/TLB/xuexinyuan/光栅/M_Decoder.mat"
Sencoder_path = r"/WORK/sunliq_work/TLB/xuexinyuan/光栅/S_Encoder.mat"
Spec_path = r"/WORK/sunliq_work/TLB/xuexinyuan/光栅/spec.mat"
SAVE_DIR = r"/WORK/sunliq_work/TLB/xuexinyuan/数据集/rcwa_iout_10k"

# ---- 读取/整理 M_Decoder ----
M_dict = hdf5storage.loadmat(M_path)
M_Decoder = M_dict["M"]
M_Decoder = torch.from_numpy(M_Decoder)
print(M_Decoder.shape)
M_Decoder = M_Decoder.permute(0, 1, 3, 2)
M_Decoder = M_Decoder.reshape(256, 256, 4 * 121)
print(M_Decoder.shape)
M_Decoder = M_Decoder.to(DEVICE).float()

# ---- 读取/整理 S_Encoder ----
Sencoder_dict = hdf5storage.loadmat(Sencoder_path)
S_Encoder = Sencoder_dict["S"]
S_Encoder = torch.from_numpy(S_Encoder)
print(S_Encoder.shape)
S_Encoder = S_Encoder.unsqueeze(4).unsqueeze(0)
print(S_Encoder.shape)
S_Encoder = S_Encoder.to(DEVICE).float()

# ---- 读取/整理 Spec ----
Spec_np = hdf5storage.loadmat(Spec_path)["spec"]              # 可能是 [121] 或 [1,121] 或更宽
Spec = np.array(Spec_np, dtype=np.float64)
Spec = torch.from_numpy(Spec)
print("Spec(raw):", Spec.shape)
# 如需裁剪频段，按你的需要解开：
# Spec = Spec[:, 15:136]
Spec = Spec.to(DEVICE).float().view(-1)  # [121] 一维

# ---- 构造 RCWA 前端 ----
rcwa = RCWAForward(
    air_path=r"/WORK/sunliq_work/TLB/xuexinyuan/光栅/Air_lambda_n_k.mat",
    si_path=r"/WORK/sunliq_work/TLB/xuexinyuan/光栅/Si_lambda_n_k.mat",
    theta_deg=45.0, phi_deg=0.0,
    device=DEVICE, dtype=None
)

# 设备 / 精度来自 rcwa（已构造）：rcwa.device, rcwa.rdtype
device = rcwa.device
dtype  = rcwa.rdtype

# 输出目录
out_dir = r"/WORK/sunliq_work/TLB/xuexinyuan/光栅/Datasets"
os.makedirs(out_dir, exist_ok=True)

# 采样规模与分片大小
N_total    = 100
shard_size = 64          # 可调：32/64/128，越大越快但更占显存

# 物理范围（单位：μm / 无量纲）
t_lo, t_hi   = 0.5, 1.0      # thickness
La_lo, La_hi = 0.1, 1.0      # Lambda
dc_lo, dc_hi = 0.1, 0.8      # duty

# Sobol 准随机采样（覆盖更均匀）
sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=12345)
U = sobol.draw(N_total).to(device=device, dtype=dtype)
thickness_all = t_lo  + (t_hi  - t_lo ) * U[:, 0]
Lambda_all    = La_lo + (La_hi - La_lo) * U[:, 1]
duty_all      = dc_lo + (dc_hi - dc_lo) * U[:, 2]

# 记录波长轴（方便下游按需使用）
lambda_list_um = torch.arange(
    rcwa.lambda_um_start,
    rcwa.lambda_um_end + 1e-12,
    rcwa.lambda_um_step,
    device=device,
    dtype=dtype,
)
L_lambda = int(lambda_list_um.numel())

# 预热一次，确认形状
with torch.no_grad():
    ms_warm = rcwa(thickness_all[0], Lambda_all[0], duty_all[0])   # [L_lambda, 16]
assert ms_warm.shape == (L_lambda, 16), f"意外的 ms 形状: {ms_warm.shape}"

# 计时
t0 = time.time()
num_shards = math.ceil(N_total / shard_size)

print(f"Start generating {N_total} samples (shard={shard_size}, device={device}) ...")

# 外层：分片进度条
for shard_idx in tqdm(range(num_shards),
                      desc=f"Shards (N={N_total}, shard={shard_size})",
                      unit="shard", dynamic_ncols=True):
    s = shard_idx * shard_size
    e = min(N_total, (shard_idx + 1) * shard_size)
    B = e - s

    # 当前分片的参数与标签
    t_batch  = thickness_all[s:e]   # [B]
    La_batch = Lambda_all[s:e]      # [B]
    dc_batch = duty_all[s:e]        # [B]
    labels   = torch.stack([t_batch, La_batch, dc_batch], dim=1)  # [B,3]

    I_out_list = []

    # 内层：样本进度条（不保留行，避免刷屏）
    with torch.no_grad():
        for i in tqdm(range(B),
                      desc=f"samples {s:05d}-{e-1:05d}",
                      unit="sample", leave=False, dynamic_ncols=True):
            ms = rcwa(t_batch[i], La_batch[i], dc_batch[i])      # [L_lambda, 16]
            ms44 = ms.reshape(L_lambda, 4, 4)                    # [L,4,4]

            I_out = compute_Iout_vectorized(ms44, S_Encoder, Spec, M_Decoder, eps=1e-12)
            if torch.is_complex(I_out):
                I_out = I_out.real
            I_out_list.append(I_out.to(dtype))

    I_out_batch = torch.stack(I_out_list, dim=0)   # [B, 256, 256]（你的 compute_Iout_vectorized 输出）

    save_payload = {
        "I_out": I_out_batch.cpu(),
        "labels": labels.cpu(),            # [B,3]：thickness(μm), Lambda(μm), duty
        "lambda_um": lambda_list_um.cpu(), # [L_lambda]
        "meta": {
            "theta_deg": rcwa.theta_deg,
            "phi_deg": rcwa.phi_deg,
            "number_of_orders": rcwa.number_of_orders,
            "dtype": str(dtype),
        }
    }

    shard_path = os.path.join(out_dir, f"shard_{s:05d}_{e-1:05d}.pt")
    torch.save(save_payload, shard_path)

print(f"Done. total_time={time.time()-t0:.1f}s, saved_dir={out_dir}")