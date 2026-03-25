from __future__ import annotations

import torch
import torch.nn as nn

from section1_params_final2 import build_section1_params_case7, real_to_complex_dtype
from section2_scan_final2 import compute_section2_results


def _run_forward_impl(config: dict, thickness_um, grating_Lambda_um, TCD_um, angle_trap_deg) -> torch.Tensor:
    para = build_section1_params_case7(
        theta_deg=config['theta_deg'],
        phi_deg=config['phi_deg'],
        grating_Lambda_um=grating_Lambda_um,
        thickness_total_um=thickness_um,
        TCD_um=TCD_um,
        angle_trap_deg=angle_trap_deg,
        number_of_layers=config['number_of_layers'],
        air_path=config['air_path'],
        si_path=config['si_path'],
        lambda_start_nm=config['lambda_start_nm'],
        lambda_stop_nm=config['lambda_stop_nm'],
        lambda_step_nm=config['lambda_step_nm'],
        device=config['device'],
        dtype=config['rdtype'],
    )
    out = compute_section2_results(para)
    return out.Muller_list[:, :, 0]


class RCWAForward(nn.Module):
    def __init__(self, air_path: str, si_path: str, theta_deg: float, phi_deg: float, device='cpu', dtype=None, auto_dtype: bool = True, number_of_layers: int = 10, lambda_start_nm: float = 480.0, lambda_stop_nm: float = 720.0, lambda_step_nm: float = 2.0):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if dtype is not None:
            self.rdtype = dtype
        elif auto_dtype:
            self.rdtype = torch.float32 if self.device.type == 'cuda' else torch.float64
        else:
            self.rdtype = torch.get_default_dtype()
        self.cdtype = real_to_complex_dtype(self.rdtype)
        self.config = {
            'air_path': air_path,
            'si_path': si_path,
            'theta_deg': theta_deg,
            'phi_deg': phi_deg,
            'device': self.device,
            'rdtype': self.rdtype,
            'number_of_layers': int(number_of_layers),
            'lambda_start_nm': float(lambda_start_nm),
            'lambda_stop_nm': float(lambda_stop_nm),
            'lambda_step_nm': float(lambda_step_nm),
        }

    def forward(self, thickness_um, grating_Lambda_um, TCD_um, angle_trap_deg) -> torch.Tensor:
        if not isinstance(thickness_um, torch.Tensor):
            thickness_um = torch.as_tensor(thickness_um, dtype=self.rdtype, device=self.device)
        else:
            thickness_um = thickness_um.to(device=self.device, dtype=self.rdtype)
        if not isinstance(grating_Lambda_um, torch.Tensor):
            grating_Lambda_um = torch.as_tensor(grating_Lambda_um, dtype=self.rdtype, device=self.device)
        else:
            grating_Lambda_um = grating_Lambda_um.to(device=self.device, dtype=self.rdtype)
        if not isinstance(TCD_um, torch.Tensor):
            TCD_um = torch.as_tensor(TCD_um, dtype=self.rdtype, device=self.device)
        else:
            TCD_um = TCD_um.to(device=self.device, dtype=self.rdtype)
        if not isinstance(angle_trap_deg, torch.Tensor):
            angle_trap_deg = torch.as_tensor(angle_trap_deg, dtype=self.rdtype, device=self.device)
        else:
            angle_trap_deg = angle_trap_deg.to(device=self.device, dtype=self.rdtype)
        return _run_forward_impl(self.config, thickness_um, grating_Lambda_um, TCD_um, angle_trap_deg)
