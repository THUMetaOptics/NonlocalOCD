# -*- coding: utf-8 -*-
"""
section1_params.py
-------------------
构建 RCWA 所需的参数对象 P（SimpleNamespace）。
- 统一 device / dtype
- 保持三大标量 grating_lambda / grating_duty_cycle / grating_thickness_total 的梯度链路
- 从 ASCII(nk) 表进行线性插值，得到随波长变化的复折射率序列 n(λ) = n - i k

用法：
from section1_params import build_section1_params
P = build_section1_params(grating_lambda=..., grating_duty_cycle=..., grating_thickness_total=...)
"""
from types import SimpleNamespace as SN
from pathlib import Path
import torch
import numpy as np

__all__ = ["build_section1_params"]

# -------------------- utils --------------------

def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


def _to_device_dtype(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _load_nk_ascii(path: str, device, dtype):
    """
    读取 ASCII nk 表（列：lambda_nm, n, k）→ torch 张量。
    （读取本身不需要梯度，可用 numpy；返回后统一到 torch 张量）
    """
    arr = np.loadtxt(Path(path).as_posix())
    xp_nm = torch.from_numpy(arr[:, 0]).to(device=device, dtype=dtype)  # 波长（nm）
    n     = torch.from_numpy(arr[:, 1]).to(device=device, dtype=dtype)
    k     = torch.from_numpy(arr[:, 2]).to(device=device, dtype=dtype)
    # 确保单调
    if not torch.all(xp_nm[1:] >= xp_nm[:-1]):
        idx = torch.argsort(xp_nm)
        xp_nm = xp_nm[idx]; n = n[idx]; k = k[idx]
    return xp_nm, n, k


@torch.no_grad()
def _ensure_monotonic_(xp: torch.Tensor):
    if xp.ndim != 1:
        raise ValueError("xp must be 1D")
    if not torch.all(xp[1:] >= xp[:-1]):
        raise ValueError("xp must be non-decreasing")


def _interp1d_linear(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    可微一维线性插值：等价 MATLAB interp1(xp, fp, x, 'linear') 的核心行为。
    边界：将 x 夹到 [xp[0], xp[-1]]，避免 NaN。
    要求：xp 递增；fp 与 xp 对应一维。
    """
    _ensure_monotonic_(xp)
    x = x.to(device=xp.device, dtype=xp.dtype)
    x = torch.clamp(x, min=xp[0], max=xp[-1])
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = torch.clamp(idx, 0, xp.numel() - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    w = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)


# -------------------- 主函数 --------------------

def build_section1_params(
    *,
    # 三个需要保留梯度链路的标量（建议外部用 nn.Parameter 传入）
    grating_lambda: torch.Tensor,          # 光栅周期（um）
    grating_duty_cycle: torch.Tensor,      # 占空比（0~1）
    grating_thickness_total: torch.Tensor, # 光栅区域总厚度（um）

    # 基本/入射设置（与你的 MATLAB 一致）
    number_of_orders: int = 41,
    lambda_um_start: float = 0.480,
    lambda_um_end: float   = 0.720,
    lambda_um_step: float  = 0.002,
    theta0_list_deg=(55.0, 65.0),
    phi0_list_deg=(0.0, 70.0),

    # 材料 nk 数据路径（ASCII：lambda[nm], n, k）
    n1_lambda_n_k_filename: str = r"F:\科研项目\XueXinyuan\2025_I2parameters\光栅\material_lambda_nk\Air_lambda_n_k.mat",
    n3_lambda_n_k_filename: str = r"F:\科研项目\XueXinyuan\2025_I2parameters\光栅\material_lambda_nk\Si_lambda_n_k.mat",
    ng_lambda_n_k_filename: str = r"F:\科研项目\XueXinyuan\2025_I2parameters\光栅\material_lambda_nk\Air_lambda_n_k.mat",
    nr_lambda_n_k_filename: str = r"F:\科研项目\XueXinyuan\2025_I2parameters\光栅\material_lambda_nk\Si_lambda_n_k.mat",

    # 光栅类型及其特参（默认：二值光栅；其余保持占位）
    grating_type: int = 0,
    number_of_layers: int = 25,
    TCD_um: float = 0.350,
    angle_trap_deg: float = 88.0,
    angle_trap_l_deg: float = 88.0,
    angle_trap_r_deg: float = 88.0,

    # 统一控制
    device="cuda",
    dtype: torch.dtype = torch.float32,
):
    """
    返回 SimpleNamespace P，含字段：
      - P.basic.number_of_orders
      - P.incident.lambda_list_um / theta0_list_deg / phi0_list_deg
      - P.grating: n1_list/n3_list/ng_list/nr_list（复数，随 λ 变化）
                   Lambda_um / thickness_total_um / duty_cycle / grating_type / ...
    """
    device = torch.device(device)
    cdtype = _complex_dtype_for(dtype)

    # 三大标量（不做 clone/detach）
    Lambda_um   = grating_lambda.to(device=device, dtype=dtype)
    Duty        = grating_duty_cycle.to(device=device, dtype=dtype)
    Thick_tot   = grating_thickness_total.to(device=device, dtype=dtype)

    # 基本设置
    P = SN()
    P.device = device
    P.dtype  = dtype
    P.cdtype = cdtype

    P.basic = SN()
    P.basic.number_of_orders = int(number_of_orders)

    # 入射波（单位：um/deg）
    lambda_list_um = torch.arange(lambda_um_start, lambda_um_end + 1e-12, lambda_um_step,
                                  device=device, dtype=dtype)
    P.incident = SN()
    P.incident.lambda_list_um  = lambda_list_um
    P.incident.theta0_list_deg = _to_device_dtype(theta0_list_deg, device, dtype)
    P.incident.phi0_list_deg   = _to_device_dtype(phi0_list_deg, device, dtype)

    # 材料 nk 插值（把查询点转 nm）
    lambda_list_nm = (lambda_list_um * 1e3).to(device=device, dtype=dtype)

    def build_n_list(nk_path: str):
        xp_nm, n, k = _load_nk_ascii(nk_path, device, dtype)
        n_interp = _interp1d_linear(xp_nm, n, lambda_list_nm)
        k_interp = _interp1d_linear(xp_nm, k, lambda_list_nm)
        return (n_interp.to(dtype) - 1j * k_interp.to(dtype)).to(dtype=cdtype)

    G = SN()
    G.n1_list = build_n_list(n1_lambda_n_k_filename)
    G.n3_list = build_n_list(n3_lambda_n_k_filename)
    G.ng_list = build_n_list(ng_lambda_n_k_filename)
    G.nr_list = build_n_list(nr_lambda_n_k_filename)

    # 几何与类型
    G.Lambda_um          = Lambda_um
    G.thickness_total_um = Thick_tot
    G.duty_cycle         = Duty
    G.grating_type       = int(grating_type)

    if G.grating_type == 0:
        pass
    elif G.grating_type == 7:
        G.number_of_layers = int(number_of_layers)
        G.TCD_um           = _to_device_dtype(TCD_um, device, dtype)
        G.angle_trap_deg   = _to_device_dtype(angle_trap_deg, device, dtype)
    elif G.grating_type == 72:
        G.number_of_layers = int(number_of_layers)
        G.TCD_um           = _to_device_dtype(TCD_um, device, dtype)
        G.angle_trap_l_deg = _to_device_dtype(angle_trap_l_deg, device, dtype)
        G.angle_trap_r_deg = _to_device_dtype(angle_trap_r_deg, device, dtype)
    else:
        G.number_of_layers = int(number_of_layers)

    P.grating = G
    return P
