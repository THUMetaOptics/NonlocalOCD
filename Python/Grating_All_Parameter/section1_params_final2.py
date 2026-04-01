from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as SN
from typing import Optional

import numpy as np
import torch


@dataclass
class NKTable:
    lambda_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray


def _resolve_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _resolve_real_dtype(dtype, device: torch.device) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.float32 if device.type == 'cuda' else torch.float64


def real_to_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise TypeError(f'Unsupported real dtype: {dtype}')


def as_real_tensor(x, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def load_nk_ascii(path: str) -> NKTable:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f'Unexpected NK file format: {path}')
    return NKTable(
        lambda_nm=np.asarray(arr[:, 0], dtype=np.float64),
        n=np.asarray(arr[:, 1], dtype=np.float64),
        k=np.asarray(arr[:, 2], dtype=np.float64),
    )


def interp_nk_to_um_grid(table: NKTable, lambda_list_um: torch.Tensor, *, device: torch.device, real_dtype: torch.dtype) -> torch.Tensor:
    x = table.lambda_nm
    n = table.n
    k = table.k
    if not np.all(np.diff(x) > 0):
        order = np.argsort(x)
        x = x[order]
        n = n[order]
        k = k[order]

    query_nm = (lambda_list_um.detach().cpu().numpy() * 1e3).astype(np.float64)
    n_q = np.interp(query_nm, x, n, left=n[0], right=n[-1])
    k_q = np.interp(query_nm, x, k, left=k[0], right=k[-1])

    n_t = torch.from_numpy(n_q).to(device=device, dtype=real_dtype)
    k_t = torch.from_numpy(k_q).to(device=device, dtype=real_dtype)
    return torch.complex(n_t, -k_t).to(real_to_complex_dtype(real_dtype))


def build_section1_params_case7(
    *,
    theta_deg,
    phi_deg,
    grating_Lambda_um,
    thickness_total_um,
    TCD_um,
    angle_trap_deg,
    number_of_layers: int = 10,
    air_path: str,
    si_path: str,
    lambda_start_nm: float = 480.0,
    lambda_stop_nm: float = 720.0,
    lambda_step_nm: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SN:
    device = _resolve_device(device or 'cpu')
    rdtype = _resolve_real_dtype(dtype, device)

    para = SN()
    para.basic = SN()
    para.basic.number_of_orders = 21

    para.incident = SN()
    para.incident.lambda_list = torch.arange(
        lambda_start_nm,
        lambda_stop_nm + 0.5 * lambda_step_nm,
        lambda_step_nm,
        device=device,
        dtype=rdtype,
    ) * 1e-3
    para.incident.theta0_list = as_real_tensor([theta_deg], device=device, dtype=rdtype)
    para.incident.phi0_list = as_real_tensor([phi_deg], device=device, dtype=rdtype)

    air_tbl = load_nk_ascii(air_path)
    si_tbl = load_nk_ascii(si_path)

    para.grating = SN()
    para.grating.n1_list = interp_nk_to_um_grid(air_tbl, para.incident.lambda_list, device=device, real_dtype=rdtype)
    para.grating.n3_list = interp_nk_to_um_grid(si_tbl, para.incident.lambda_list, device=device, real_dtype=rdtype)
    para.grating.ng_list = interp_nk_to_um_grid(air_tbl, para.incident.lambda_list, device=device, real_dtype=rdtype)
    para.grating.nr_list = interp_nk_to_um_grid(si_tbl, para.incident.lambda_list, device=device, real_dtype=rdtype)

    para.grating.grating_type = 7
    para.grating.Lambda = as_real_tensor(grating_Lambda_um, device=device, dtype=rdtype).reshape(())
    para.grating.thickness_total = as_real_tensor(thickness_total_um, device=device, dtype=rdtype).reshape(())
    para.grating.number_of_layers = int(number_of_layers)
    para.grating.TCD = as_real_tensor(TCD_um, device=device, dtype=rdtype).reshape(())
    para.grating.angle_trap = as_real_tensor(angle_trap_deg, device=device, dtype=rdtype).reshape(())

    return para
