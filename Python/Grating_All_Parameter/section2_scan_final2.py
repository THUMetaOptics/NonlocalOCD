from __future__ import annotations

from types import SimpleNamespace as SN

import torch

from polarimetry_final2 import cal_Jones_and_Muller_matrix_useTMTE_0order
from rcwa_core_final2 import simu_rcwa1d_main


def compute_section2_results(para: SN) -> SN:
    dev = para.grating.n1_list.device
    rd = para.incident.lambda_list.dtype
    cd = para.grating.n1_list.dtype

    L_theta0 = int(para.incident.theta0_list.numel())
    L_lambda = int(para.incident.lambda_list.numel())

    Muller_list = torch.zeros((L_lambda, 16, L_theta0), dtype=rd, device=dev)
    r_sp_list = torch.zeros((L_lambda, 1, L_theta0), dtype=cd, device=dev)
    r_pp_list = torch.zeros((L_lambda, 1, L_theta0), dtype=cd, device=dev)
    r_ss_list = torch.zeros((L_lambda, 1, L_theta0), dtype=cd, device=dev)
    r_ps_list = torch.zeros((L_lambda, 1, L_theta0), dtype=cd, device=dev)
    Psi_Delta_list = torch.zeros((L_lambda, 6, L_theta0), dtype=rd, device=dev)

    tiny = torch.as_tensor(1e-30, device=dev, dtype=rd)

    for kk1 in range(L_theta0):
        para.incident.theta0 = para.incident.theta0_list[kk1]
        para.incident.phi0 = para.incident.phi0_list[kk1]
        for kk2 in range(L_lambda):
            para.incident.lambda_ = para.incident.lambda_list[kk2]
            setattr(para.incident, 'lambda', para.incident.lambda_)
            para.grating.n1 = para.grating.n1_list[kk2]
            para.grating.n3 = para.grating.n3_list[kk2]
            para.grating.ng = para.grating.ng_list[kk2]
            para.grating.nr = para.grating.nr_list[kk2]

            para.incident.psi0 = torch.as_tensor(0.0, device=dev, dtype=rd)
            Rsp0, Rpp0 = simu_rcwa1d_main(para)

            para.incident.psi0 = torch.as_tensor(90.0, device=dev, dtype=rd)
            Rss0, Rps0 = simu_rcwa1d_main(para)

            _, Muller_vector = cal_Jones_and_Muller_matrix_useTMTE_0order(Rsp0, Rpp0, Rss0, Rps0, para.grating.n1)

            def safe_div(a, b):
                b_safe = torch.where(torch.abs(b) < tiny, b + tiny.to(dtype=b.dtype), b)
                return a / b_safe

            r_pp_ss = safe_div(Rpp0, Rss0)
            r_ps_ss = safe_div(Rps0, Rss0)
            r_sp_ss = safe_div(Rsp0, Rss0)

            Psi_pp = torch.rad2deg(torch.atan(torch.abs(r_pp_ss))).real
            Delta_pp = torch.rad2deg(torch.angle(r_pp_ss)).real
            Psi_ps = torch.rad2deg(torch.atan(torch.abs(r_ps_ss))).real
            Delta_ps = torch.rad2deg(torch.angle(r_ps_ss)).real
            Psi_sp = torch.rad2deg(torch.atan(torch.abs(r_sp_ss))).real
            Delta_sp = torch.rad2deg(torch.angle(r_sp_ss)).real

            Muller_list[kk2, :, kk1] = Muller_vector
            r_sp_list[kk2, 0, kk1] = Rsp0
            r_pp_list[kk2, 0, kk1] = Rpp0
            r_ss_list[kk2, 0, kk1] = Rss0
            r_ps_list[kk2, 0, kk1] = Rps0
            Psi_Delta_list[kk2, :, kk1] = torch.stack([
                Psi_pp, Delta_pp, Psi_ps, Delta_ps, Psi_sp, Delta_sp
            ], dim=0).to(rd)

    out = SN()
    out.Muller_list = Muller_list
    out.r_sp_list = r_sp_list
    out.r_pp_list = r_pp_list
    out.r_ss_list = r_ss_list
    out.r_ps_list = r_ps_list
    out.Psi_Delta_list = Psi_Delta_list
    return out
