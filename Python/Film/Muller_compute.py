# -*- coding: utf-8 -*-
# @Author : PopSama
# @Time : 2025-09-11 9:10


import torch
import torch.nn as nn
import numpy as np


class MuellerMatrixFilmSingleLayer(nn.Module):
    """
    单层薄膜 Mueller 矩阵（批量），n1(λ)=A + B/λ^2 + C/λ^4（λ单位 μm）。
    Popsama额外说明，如果之后更换折射率模型（即上面的公式）整个模型需要重写！！！！
    __init__ 里只存默认 A/B/C；真正的 n1(λ) 在 forward 中由传入的 A/B/C 即时计算，
    因此可将 NN 输出的 A/B 直接作为 forward 参数传入，梯度可回传到 NN。
    """

    def __init__(self, si_re_path, si_im_path, device="cpu",
                 A: float = 1.4721, B: float = 0.0338, C: float = 0.0):
        super().__init__()
        self.device = torch.device(device)

        # 读入硅的 n,k（文件里 λ 单位需与你数据一致；此处沿用你之前的实现）
        si_re_np = np.genfromtxt(si_re_path, delimiter=",",
                                 converters={0: self._safe_float, 1: self._safe_float})
        si_re_np = np.flip(si_re_np, axis=0).copy()

        si_im_np = np.genfromtxt(si_im_path, delimiter=",",
                                 converters={0: self._safe_float, 1: self._safe_float})
        si_im_np = np.flip(si_im_np, axis=0).copy()

        lambda_t = torch.from_numpy(si_re_np[:, 0]).to(self.device, dtype=torch.float64)  # μm
        n_re_t   = torch.from_numpy(si_re_np[:, 1]).to(self.device, dtype=torch.float64)
        n_im_t   = torch.from_numpy(si_im_np[:, 1]).to(self.device, dtype=torch.float64)

        # 常量/表格做成 buffer
        self.register_buffer('lambda_', lambda_t)  # [L], μm
        self.register_buffer('n0_', torch.ones_like(n_re_t, dtype=torch.complex128))  # 空气
        self.register_buffer('n2_', (n_re_t + 1j * n_im_t).to(torch.complex128))      # 硅
        self.register_buffer('lambda0', torch.linspace(480., 720., 121, device=self.device, dtype=torch.float64))  # 目标 nm 轴

        # 默认 A/B/C（可被 forward 覆写）
        self.register_buffer('A_default', torch.tensor(A, device=self.device, dtype=torch.float64))
        self.register_buffer('B_default', torch.tensor(B, device=self.device, dtype=torch.float64))
        self.register_buffer('C_default', torch.tensor(C, device=self.device, dtype=torch.float64))

    # ------- 工具 -------
    def _safe_float(self, x):
        s = x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else str(x)
        return float(s.strip().rstrip(','))

    # ------- 前向：可选 A/B/C 覆写 -------
    def _ensure_tensor64(self, t):
        # 不要 @torch.no_grad()
        if isinstance(t, torch.Tensor):
            # 这一步是可导的；不要用 no_grad 包裹
            return t.to(self.lambda_.device, dtype=torch.float64)
        else:
            # 只有在确实是 Python 标量/float 时才新建 Tensor
            return torch.as_tensor(t, device=self.lambda_.device, dtype=torch.float64)

    def forward(self, d, theta, A=None, B=None, C=None):

        rd = torch.float64; cd = torch.complex128

        # --- 输入整理（确保不切断图）---
        if isinstance(d, torch.Tensor):
            d = d.to(self.device, rd)
        else:
            d = torch.tensor(d, device=self.device, dtype=rd)

        if isinstance(theta, torch.Tensor):
            theta = theta.to(self.device, rd)
        else:
            theta = torch.tensor(theta, device=self.device, dtype=rd)

        if d.dim() == 1:     d = d.unsqueeze(-1)
        if theta.dim() == 1: theta = theta.unsqueeze(-1)

        Bsz = d.shape[0]
        lam = self.lambda_               # [L], μm
        L   = lam.shape[0]

        # --- A/B/C 处理：支持标量或按样本 ---
        # 不能用 no_grad 包裹整个块，否则会截断梯度；仅把常量搬 dtype/device 时用了上面的辅助函数
        A = self._ensure_tensor64(A if A is not None else self.A_default)
        B = self._ensure_tensor64(B if B is not None else self.B_default)
        C = self._ensure_tensor64(C if C is not None else self.C_default)

        # 形状整理为 [B,1]，以便与 λ 轴广播
        if A.dim() == 0: A = A.view(1,1).expand(Bsz, 1)
        elif A.dim() == 1: A = A.view(-1,1)
        if B.dim() == 0: B = B.view(1,1).expand(Bsz, 1)
        elif B.dim() == 1: B = B.view(-1,1)
        if C.dim() == 0: C = C.view(1,1).expand(Bsz, 1)
        elif C.dim() == 1: C = C.view(-1,1)

        # --- n1(λ) = A + B/λ^2 + C/λ^4 → [B,L] ---
        inv_lam2 = (1.0 / (lam * lam)).unsqueeze(0)        # [1,L]
        inv_lam4 = (inv_lam2 * inv_lam2)                   # [1,L]
        n1_real  = A + B * inv_lam2 + C * inv_lam4         # [B,1] + [B,1]*[1,L] + ...
        n1_b     = n1_real.to(cd).expand(-1, L)            # [B,L] complex128

        # --- 其他介质扩展到批次 ---
        n0_b = self.n0_.unsqueeze(0).expand(Bsz, L)        # [B,L]
        n2_b = self.n2_.unsqueeze(0).expand(Bsz, L)        # [B,L]

        # --- 光学计算 ---
        d_mm   = d * 1e-3                                  # um → mm
        theta_rad = theta * (torch.pi / 180.)
        phi0   = theta_rad.expand(Bsz, L).to(cd)

        # 斯涅尔（用 real）
        phi1 = torch.asin(n0_b.real * torch.sin(phi0) / n1_b.real)
        phi2 = torch.asin(n1_b.real * torch.sin(phi1) / n2_b.real)

        cos0, cos1, cos2 = torch.cos(phi0), torch.cos(phi1), torch.cos(phi2)

        # 顶/底界面 Fresnel
        r1p = (n1_b * cos0 - n0_b * cos1) / (n1_b * cos0 + n0_b * cos1)
        r1s = (n0_b * cos0 - n1_b * cos1) / (n0_b * cos0 + n1_b * cos1)

        Rp_bottom = (n2_b * cos1 - n1_b * cos2) / (n2_b * cos1 + n1_b * cos2)
        Rs_bottom = (n1_b * cos1 - n2_b * cos2) / (n1_b * cos1 + n2_b * cos2)

        # 干涉相位（注意单位）
        d_bL   = d_mm.expand(Bsz, L)
        delta  = 2. * torch.pi * (d_bL / lam.unsqueeze(0)) * n1_b.real * cos1
        exp_t  = torch.exp(-2j * delta)

        Rp = (r1p + Rp_bottom * exp_t) / (1. + r1p * Rp_bottom * exp_t)
        Rs = (r1s + Rs_bottom * exp_t) / (1. + r1s * Rs_bottom * exp_t)

        M_temp   = self._calculate_mueller_matrix(Rp, Rs, Bsz, L)            # [B,L,4,4]
        M_interp = self._interpolate_to_standard_wavelengths(M_temp, lam.unsqueeze(0).expand(Bsz, L), Bsz)
        return M_interp  # [B,4,4,121]

    # ------- 计算 Mueller -------
    def _calculate_mueller_matrix(self, Rpp, Rss, B, L):
        Rps = torch.zeros_like(Rpp)
        Rsp = torch.zeros_like(Rpp)

        M = torch.zeros((B, L, 4, 4), dtype=torch.float64, device=self.device)

        Rpp2, Rss2 = torch.abs(Rpp)**2, torch.abs(Rss)**2
        Rps2, Rsp2 = torch.abs(Rps)**2, torch.abs(Rsp)**2

        Rpp_Rps_c = Rpp * torch.conj(Rps)
        Rsp_Rss_c = Rsp * torch.conj(Rss)
        Rpp_Rsp_c = Rpp * torch.conj(Rsp)
        Rps_Rss_c = Rps * torch.conj(Rss)
        Rpp_Rss_c = Rpp * torch.conj(Rss)
        Rps_Rsp_c = Rps * torch.conj(Rsp)

        # Row 1
        M[..., 0, 0] = 0.5 * (Rpp2 + Rsp2 + Rps2 + Rss2)
        M[..., 0, 1] = 0.5 * (Rpp2 + Rsp2 - Rps2 - Rss2)
        M[..., 0, 2] = -(Rpp_Rps_c + Rsp_Rss_c).real
        M[..., 0, 3] = -(Rpp_Rps_c + Rsp_Rss_c).imag
        # Row 2
        M[..., 1, 0] = 0.5 * (Rpp2 - Rsp2 + Rps2 - Rss2)
        M[..., 1, 1] = 0.5 * (Rpp2 - Rsp2 - Rps2 + Rss2)
        M[..., 1, 2] =  (Rpp_Rps_c - Rsp_Rss_c).real
        M[..., 1, 3] =  (Rpp_Rps_c - Rsp_Rss_c).imag
        # Row 3
        M[..., 2, 0] = -(Rpp_Rsp_c + Rps_Rss_c).real
        M[..., 2, 1] =  (Rpp_Rsp_c - Rps_Rss_c).real
        M[..., 2, 2] =  (Rpp_Rss_c + Rps_Rsp_c).real
        M[..., 2, 3] = -(Rpp_Rss_c - Rps_Rsp_c).imag
        # Row 4
        M[..., 3, 0] =  (Rpp_Rsp_c + Rps_Rss_c).imag
        M[..., 3, 1] = -(Rpp_Rsp_c - Rps_Rss_c).imag
        M[..., 3, 2] =  (Rpp_Rss_c + Rps_Rsp_c).imag
        M[..., 3, 3] =  (Rpp_Rss_c - Rps_Rsp_c).real
        return M

    # ------- 插值到 480–720 nm（121 点）-------
    def _interpolate_to_standard_wavelengths(self, M_temp, lambda_b, B):
        L = lambda_b.shape[1]
        M_out = torch.zeros((B, 4, 4, 121), dtype=torch.float64, device=self.device)

        lam_nm = lambda_b * 1e3  # μm -> nm
        for i in range(121):
            lam0 = self.lambda0[i].item()
            diff = (lam_nm - lam0).abs()                # [B,L]
            mask = diff <= 0.3                          # 容差 0.3 nm
            mask_rev = torch.flip(mask, dims=[1])
            idx_rev  = torch.argmax(mask_rev.to(torch.int32), dim=1)
            j_idx    = (L - 1) - idx_rev

            for r in range(4):
                for c in range(4):
                    M_out[:, r, c, i] = M_temp[torch.arange(B, device=self.device), j_idx, r, c]
        return M_out
