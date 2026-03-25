# -*- coding: utf-8 -*-
"""
section2_scan.py  (v1.0.1)
-----------------
扫描 θ、λ，调用 RCWA 主流程，返回：
- Muller_list: [L_lambda, 16, L_theta]
- r_sp_list, r_pp_list, r_ss_list, r_ps_list: [L_lambda, 1, L_theta]
- Psi_Delta_list: [L_lambda, 6, L_theta]

内部包含：
- simu_rcwa1d_main_torch（外层薄封装）
- rcwa_procedure_torch（RCWA 核心）
- jones_muller_torch（由 0 级反射系数构造 Jones/Mueller）

更新：修复 s_1 线性方程组处多余右括号导致的语法问题。

用法：
from section2_scan import compute_section2_results
res = compute_section2_results(P)
"""
from types import SimpleNamespace as SN
import torch


def _as_1d_tensor(x, device, dtype):
    """把 tuple/list/tensor 统一成 1D torch.Tensor（保 device/dtype）"""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).reshape(-1)
    # 允许 None -> 报错更清楚
    if x is None:
        raise ValueError("Expected a list/tensor, got None.")
    return torch.as_tensor(x, device=device, dtype=dtype).reshape(-1)


__all__ = [
    "compute_section2_results",
    # 可选导出：
    "simu_rcwa1d_main_torch",
    "rcwa_procedure_torch",
    "jones_muller_torch",
]

# -------------------- helpers --------------------

def _to_device_dtype(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _cdiv(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """复数安全相除：a/b = a*conj(b)/(abs(b)^2 + eps)。"""
    return (a * torch.conj(b)) / (torch.abs(b) ** 2 + eps)


def _flatten_muller(M: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if M.numel() == 16 and (M.ndim == 1 or M.shape == (16,)):
        v = M
    else:
        if M.ndim == 2 and M.shape == (4, 4):
            v = M.reshape(-1)  # MATLAB: M' 然后(:)
        else:
            raise ValueError(f"Unexpected Muller shape {tuple(M.shape)}; expect [4,4] or [16].")
    return v.real.to(dtype)


def _diag(v):
    return torch.diag_embed(v)


def _toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Toeplitz 矩阵（兼容 PyTorch，不用负步长切片）"""
    if c.ndim != 1 or r.ndim != 1:
        raise ValueError("toeplitz expects 1D tensors for c and r")

    c = c.reshape(-1)
    r = r.reshape(-1).clone()
    if r.numel() > 0 and c.numel() > 0:
        r[0] = c[0]
    p = int(r.numel())  # first row length
    m = int(c.numel())  # first column length
    left = r.new_empty((0,), dtype=r.dtype, device=r.device) if p <= 1 else torch.flip(r[1:], dims=[0])
    x = torch.cat([left, c], dim=0)
    i = torch.arange(m, device=c.device)
    j = torch.arange(p, 0, -1, device=c.device)  # p ... 1
    ij = i.unsqueeze(1) + j.unsqueeze(0) - 1     # 0-based 索引
    t = x[ij]
    return t


# -------------------- Jones & Mueller --------------------

def jones_muller_torch(Rsp0: torch.Tensor,
                       Rpp0: torch.Tensor,
                       Rss0: torch.Tensor,
                       Rps0: torch.Tensor,
                       n1:   torch.Tensor):
    """等价 cal_Jones_and_Muller_matrix_useTMTE_0order。"""
    device = Rsp0.device
    cdtype = Rsp0.dtype

    # Jones（douch 0 映射）
    J11 = Rpp0  # r_pp
    J12 = Rps0  # r_ps
    J21 = Rsp0  # r_sp
    J22 = Rss0  # r_ss
    Jones_matrix = torch.stack([torch.stack([J11, J12], dim=0),
                                torch.stack([J21, J22], dim=0)], dim=0)

    # Mueller
    U = torch.tensor([[1, 0, 0, 1],
                      [1, 0, 0,-1],
                      [0, 1, 1, 0],
                      [0, 1j,-1j,0]], device=device, dtype=cdtype)
    K = torch.kron(Jones_matrix, torch.conj(Jones_matrix))
    U_inv = torch.linalg.inv(U)
    M = U @ K @ U_inv
    M = M / M[0, 0]
    Muller_vec = M.reshape(-1)  # 展平为[16]
    return Jones_matrix, Muller_vec


# -------------------- RCWA 核心 --------------------

def rcwa_procedure_torch(ctx: SN) -> SN:
    d = ctx.device
    rdtype = ctx.dtype
    cdtype = ctx.cdtype

    number_of_orders = int(ctx.number_of_orders)
    M = number_of_orders - 1
    order_max = (number_of_orders - 1) // 2
    n_vec = torch.arange(-order_max, order_max + 1, device=d, dtype=rdtype)

    I = torch.eye(number_of_orders, device=d, dtype=cdtype)
    zero = torch.zeros((number_of_orders, number_of_orders), device=d, dtype=cdtype)

    theta = ctx.theta0_deg * torch.pi / 180.0
    phi   = ctx.phi0_deg   * torch.pi / 180.0
    psi   = (ctx.psi0_deg * torch.pi / 180.0) if (ctx.polarization == 0) else torch.tensor(0.0, device=d, dtype=rdtype)
    theta = torch.where(theta == 0, torch.tensor(1e-20, device=d, dtype=rdtype), theta)

    k_0 = 2.0 * torch.pi / (ctx.lambda_um * 1e-6)
    K   = 2.0 * torch.pi / (ctx.Lambda_um * 1e-6)

    k_x = k_0 * ctx.n1 * torch.sin(theta) * torch.cos(phi) + n_vec.to(cdtype) * K
    k_x_2 = k_x * k_x
    K_x   = _diag((k_x / k_0).to(cdtype))
    K_x_2 = K_x @ K_x

    k_y = (k_0 * ctx.n1 * torch.sin(theta) * torch.sin(phi)).to(cdtype)
    k_y_2 = k_y * k_y
    k_y_2_I = (k_y_2.real.to(rdtype)) * torch.eye(number_of_orders, device=d, dtype=rdtype)
    k_y_2_I = k_y_2_I.to(cdtype)

    k_y_k_0 = (k_y / k_0).to(cdtype).real.to(rdtype)

    if torch.abs(k_y) == 0:
        phi_i = torch.zeros(number_of_orders, device=d, dtype=rdtype)
    else:
        phi_i = torch.atan((k_y.real / (k_x.real + 1e-30)).to(rdtype))

    # 区域1/3的 k_z
    k1_abs = k_0 * torch.abs(ctx.n1)
    k3_abs = k_0 * torch.abs(ctx.n3)
    base = (k_x_2.real + k_y_2.real).to(rdtype)
    k_1_z = []
    k_3_z = []
    for m in range(number_of_orders):
        val = torch.sqrt(base[m])
        kz1 = torch.where(
            val < k1_abs,
            torch.sqrt((k_0 * ctx.n1) * (k_0 * ctx.n1) - (k_x_2[m] + k_y_2)),
            -1j * torch.sqrt((k_x_2[m] + k_y_2) - (k_0 * ctx.n1) * (k_0 * ctx.n1)),
        )
        k_1_z.append(kz1)
        kz3 = torch.where(
            val < k3_abs,
            torch.sqrt((k_0 * ctx.n3) * (k_0 * ctx.n3) - (k_x_2[m] + k_y_2)),
            -1j * torch.sqrt((k_x_2[m] + k_y_2) - (k_0 * ctx.n3) * (k_0 * ctx.n3)),
        )
        k_3_z.append(kz3)
    k_1_z = torch.stack(k_1_z).to(cdtype)
    k_3_z = torch.stack(k_3_z).to(cdtype)

    k_1_z_n = torch.where(k_1_z.imag == 0, -k_1_z, k_1_z)
    k_3_z_n = torch.where(k_3_z.imag == 0, -k_3_z, k_3_z)

    # S_matrix_initialization
    F_c = _diag(torch.cos(phi_i).to(cdtype))
    F_s = _diag(torch.sin(phi_i).to(cdtype))

    k_1_z_k_0 = (k_1_z / k_0).to(cdtype)
    Y1 = _diag(k_1_z_k_0)
    Z1 = _diag(k_1_z_k_0 / (ctx.n1 * ctx.n1))

    k_3_z_k_0 = (k_3_z / k_0).to(cdtype)
    Y3 = _diag(k_3_z_k_0)
    Z3 = _diag(k_3_z_k_0 / (ctx.n3 * ctx.n3))

    # 每层求解（一般锥形）
    L = int(ctx.number_of_layers)
    Vss = [None] * L; Wss = [None] * L; Vsp = [None] * L; Wsp = [None] * L
    Wpp = [None] * L; Vpp = [None] * L; Wps = [None] * L; Vps = [None] * L
    Q1  = [None] * L; W1  = [None] * L
    Q2  = [None] * L; W2  = [None] * L

    M2 = M
    mm = torch.arange(1, M2 + 1, device=d, dtype=rdtype)
    neg_to_pos = torch.arange(-M2, M2 + 1, device=d, dtype=rdtype)

    def _get_layer_scalar(x, idx):
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return x.to(device=d, dtype=rdtype)
            elif x.numel() == L:
                return x.reshape(-1)[idx].to(device=d, dtype=rdtype)
            else:
                return x.to(device=d, dtype=rdtype)
        else:
            return torch.tensor(float(x), device=d, dtype=rdtype)

    for l in range(L - 1, -1, -1):
        epsg = (ctx.ng ** 2).to(cdtype)
        epsr = (ctx.nr ** 2).to(cdtype)

        duty_l  = _get_layer_scalar(ctx.duty_cycle, l)
        shift_l = _get_layer_scalar(ctx.shift, l)

        epsG   = (1 - duty_l) * epsg + duty_l * epsr
        i_epsG = (1 - duty_l) / epsg + duty_l / epsr

        Sinc = torch.sin(torch.pi * duty_l * mm) / (torch.pi * mm)
        v_m  = (epsr - epsg) * torch.flip(Sinc, dims=[0])
        v_0  = epsG
        v_p  = (epsr - epsg) * Sinc
        phase = torch.exp(-1j * 2 * torch.pi * shift_l * neg_to_pos)
        v = torch.cat([v_m, v_0.unsqueeze(0), v_p], dim=0) * phase

        i_vm = (1/epsr - 1/epsg) * torch.flip(Sinc, dims=[0])
        i_v0 = i_epsG
        i_vp = (1/epsr - 1/epsg) * Sinc
        i_v = torch.cat([i_vm, i_v0.unsqueeze(0), i_vp], dim=0) * phase

        v1 = torch.flip(v[:number_of_orders], dims=[0]).to(cdtype)
        v2 = v[number_of_orders-1 : 2*number_of_orders-1].to(cdtype)
        E_l = _toeplitz(v1, v2)

        i_v1 = torch.flip(i_v[:number_of_orders], dims=[0]).to(cdtype)
        i_v2 = i_v[number_of_orders-1 : 2*number_of_orders-1].to(cdtype)
        A_l  = _toeplitz(i_v1, i_v2)

        eigen1 = (k_y_2_I / (k_0 * k_0)) + K_x_2 - E_l
        evals1, evecs1 = torch.linalg.eig(eigen1)
        Q1_l = _diag(torch.sqrt(evals1))
        W1_l = evecs1
        Q1[l] = Q1_l; W1[l] = W1_l

        inv_E_l = torch.linalg.inv(E_l)
        A = K_x_2 - E_l
        B = K_x @ inv_E_l @ K_x - I
        if int(ctx.faktorization) == 1:
            eigen2 = (k_y_2_I / (k_0 * k_0)) + B @ torch.linalg.inv(A_l)
        else:
            eigen2 = (k_y_2_I / (k_0 * k_0)) + B @ E_l
        evals2, evecs2 = torch.linalg.eig(eigen2)
        Q2_l = _diag(torch.sqrt(evals2))
        W2_l = evecs2
        Q2[l] = Q2_l; W2[l] = W2_l

        inv_A = torch.linalg.inv(A)
        inv_B = torch.linalg.inv(B)

        V11 = inv_A @ W1_l @ Q1_l
        V12 = k_y_k_0 * (inv_A @ K_x @ W2_l).to(cdtype)
        V21 = k_y_k_0 * (inv_B @ K_x @ inv_E_l @ W1_l).to(cdtype)
        V22 = inv_B @ W2_l @ Q2_l

        Vss_l = F_c @ V11
        Wss_l = F_c @ W1_l + F_s @ V21
        Vsp_l = F_c @ V12 - F_s @ W2_l
        Wsp_l = F_s @ V22
        Wpp_l = F_c @ V22
        Vpp_l = F_c @ W2_l + F_s @ V12
        Wps_l = F_c @ V21 - F_s @ W1_l
        Vps_l = F_s @ V11

        Vss[l] = Vss_l; Wss[l] = Wss_l; Vsp[l] = Vsp_l; Wsp[l] = Wsp_l
        Wpp[l] = Wpp_l; Vpp[l] = Vpp_l; Wps[l] = Wps_l; Vps[l] = Vps_l

    # 组装 Li 的 S 矩阵
    W_3_11 = torch.cat([torch.cat([I,           zero], dim=1),
                        torch.cat([-1j*Y1,      zero], dim=1)], dim=0)
    W_3_12 = torch.cat([torch.cat([ torch.sin(psi)*I,                   zero], dim=1),
                        torch.cat([ 1j*ctx.n1*torch.cos(theta)*torch.sin(psi)*I, zero], dim=1)], dim=0)
    W_3_21 = torch.cat([torch.cat([zero,        I   ], dim=1),
                        torch.cat([zero,       -1j*Z1], dim=1)], dim=0)
    W_3_22 = torch.cat([torch.cat([-1j*ctx.n1*torch.cos(psi)*I,        zero], dim=1),
                        torch.cat([ torch.cos(psi)*torch.cos(theta)*I, zero], dim=1)], dim=0)

    W_2_11 = [None]*L; W_2_12=[None]*L; W_2_21=[None]*L; W_2_22=[None]*L
    for i in range(L):
        W_2_11[i] = torch.cat([torch.cat([ Vss[i],  Vsp[i]], dim=1),
                               torch.cat([-Wss[i], -Wsp[i]], dim=1)], dim=0)
        W_2_12[i] = torch.cat([torch.cat([ Vss[i],  Vsp[i]], dim=1),
                               torch.cat([ Wss[i],  Wsp[i]], dim=1)], dim=0)
        W_2_21[i] = torch.cat([torch.cat([-Wps[i], -Wpp[i]], dim=1),
                               torch.cat([ Vps[i],  Vpp[i]], dim=1)], dim=0)
        W_2_22[i] = torch.cat([torch.cat([ Wps[i],  Wpp[i]], dim=1),
                               torch.cat([ Vps[i],  Vpp[i]], dim=1)], dim=0)

    W_1_11 = torch.zeros_like(W_3_11)
    W_1_12 = torch.cat([torch.cat([I,   torch.zeros_like(I)], dim=1),
                        torch.cat([1j*Y3, torch.zeros_like(I)], dim=1)], dim=0)
    W_1_21 = torch.zeros_like(W_3_11)
    W_1_22 = torch.cat([torch.cat([torch.zeros_like(I), I ], dim=1),
                        torch.cat([torch.zeros_like(I), 1j*Z3], dim=1)], dim=0)

    s_2 = torch.linalg.solve(torch.cat([torch.cat([ W_3_11,          -W_2_12[0]], dim=1),
                                        torch.cat([ W_3_21,          -W_2_22[0]], dim=1)], dim=0),
                             torch.cat([torch.cat([ W_2_11[0],       -W_3_12],    dim=1),
                                        torch.cat([ W_2_21[0],       -W_3_22],    dim=1)], dim=0))

    s_1 = [None]*(L-1) if L>1 else []
    for i in range(L-1):
        s_1[L-2-i] = torch.linalg.solve(
            torch.cat([torch.cat([ W_2_11[i],          -W_2_12[i+1]], dim=1),
                       torch.cat([ W_2_21[i],          -W_2_22[i+1]], dim=1)], dim=0),
            torch.cat([torch.cat([ W_2_11[i+1],        -W_2_12[i]],   dim=1),
                       torch.cat([ W_2_21[i+1],        -W_2_22[i]],   dim=1)], dim=0)
        )

    s_0 = torch.linalg.solve(torch.cat([torch.cat([ W_2_11[L-1],      -W_1_12], dim=1),
                                        torch.cat([ W_2_21[L-1],      -W_1_22], dim=1)], dim=0),
                             torch.cat([torch.cat([ W_1_11,           -W_2_12[L-1]], dim=1),
                                        torch.cat([ W_1_21,           -W_2_22[L-1]], dim=1)], dim=0))

    X_1 = [None]*L; X_2=[None]*L
    for i in range(L):
        q1 = torch.diagonal(Q1[i], dim1=-2, dim2=-1)
        q2 = torch.diagonal(Q2[i], dim1=-2, dim2=-1)
        X_1[L-1-i] = _diag(torch.exp(-k_0 * q1 * ctx.layer_thickness_m))
        X_2[L-1-i] = _diag(torch.exp(-k_0 * q2 * ctx.layer_thickness_m))

    a_ud = [None]*(L+1)
    a_dd = [None]*(L+1)
    a_ud[0] = s_0[:2*number_of_orders, 2*number_of_orders:]
    a_dd[0] = s_0[2*number_of_orders:, 2*number_of_orders:]

    new_I = torch.eye(2*number_of_orders, device=d, dtype=cdtype)

    for i in range(L):
        if i == L-1:
            first_m  = torch.block_diag(I, I, X_1[L-1], X_2[L-1])
            second_m = torch.block_diag(X_1[L-1], X_2[L-1], I, I)
            s_2_l = first_m @ s_2 @ second_m
            b_uu = s_2_l[:2*number_of_orders, :2*number_of_orders]
            b_ud = s_2_l[:2*number_of_orders, 2*number_of_orders:]
            b_du = s_2_l[2*number_of_orders:, :2*number_of_orders]
            b_dd = s_2_l[2*number_of_orders:, 2*number_of_orders:]
        else:
            first_m  = torch.block_diag(I, I, X_1[i], X_2[i])
            second_m = torch.block_diag(X_1[i], X_2[i], I, I)
            s_1_l = first_m @ s_1[i] @ second_m
            b_uu = s_1_l[:2*number_of_orders, :2*number_of_orders]
            b_ud = s_1_l[:2*number_of_orders, 2*number_of_orders:]
            b_du = s_1_l[2*number_of_orders:, :2*number_of_orders]
            b_dd = s_1_l[2*number_of_orders:, 2*number_of_orders:]

        inv_mid = torch.linalg.inv(new_I - b_du @ a_ud[i])
        a_ud[i+1] = b_ud + b_uu @ (a_ud[i] @ inv_mid) @ b_dd
        a_dd[i+1] = (a_dd[i] @ inv_mid) @ b_dd

    T_dd = a_dd[L]
    R_ud = a_ud[L]

    d_n1 = torch.zeros((2*number_of_orders, 1), device=d, dtype=cdtype)
    d_n1[(number_of_orders-1)//2, 0] = 1.0

    R = R_ud @ d_n1
    # T = T_dd @ d_n1  # 若后续需要可用

    R_s = R[:number_of_orders, 0]
    R_p = R[number_of_orders:, 0]
    return SN(R_s=R_s, R_p=R_p)


# -------------------- RCWA 外层薄封装 --------------------

def simu_rcwa1d_main_torch(para: SN, *, rcwa_procedure_fn=rcwa_procedure_torch):
    device = para.grating.n1.device
    dtype  = torch.float32 if para.grating.n1.dtype == torch.complex64 else torch.float64
    cdtype = para.grating.n1.dtype

    number_of_orders    = int(para.basic.number_of_orders)
    matrix_algorithm    = 1
    change_matrix_base  = 1
    use_dispersion      = 2
    polarization        = 0
    faktorization       = 1

    lam_attr = getattr(para.incident, "lambda", None)
    if lam_attr is None:
        lam_um = para.incident.lambda_um.to(device=device, dtype=dtype)
    else:
        lam_um = lam_attr.to(device=device, dtype=dtype)
    theta0  = para.incident.theta0.to(device=device, dtype=dtype)
    phi0    = para.incident.phi0.to(device=device, dtype=dtype)
    psi0    = para.incident.psi0.to(device=device, dtype=dtype)

    if hasattr(para.grating, "Lambda"):
        Lambda_um = para.grating.Lambda.to(device=device, dtype=dtype)
    else:
        Lambda_um = para.grating.Lambda_um.to(device=device, dtype=dtype)
        para.grating.Lambda = Lambda_um

    thickness_total_um = para.grating.thickness_total if hasattr(para.grating, "thickness_total") else para.grating.thickness_total_um
    thickness_total_um = thickness_total_um.to(device=device, dtype=dtype)

    duty_cycle   = para.grating.duty_cycle.to(device=device, dtype=dtype)
    grating_type = int(para.grating.grating_type)

    n1 = para.grating.n1.to(dtype=cdtype)
    n3 = para.grating.n3.to(dtype=cdtype)
    ng = para.grating.ng.to(dtype=cdtype)
    nr = para.grating.nr.to(dtype=cdtype)

    thickness_m      = thickness_total_um * 1e-6
    number_of_layers = 1
    layer_thickness  = thickness_m
    shift            = torch.tensor(0.5, device=device, dtype=dtype)

    ctx = SN(
        number_of_orders=number_of_orders,
        matrix_algorithm=matrix_algorithm,
        change_matrix_base=change_matrix_base,
        use_dispersion=use_dispersion,
        polarization=polarization,
        faktorization=faktorization,
        lambda_um=lam_um,
        theta0_deg=theta0,
        phi0_deg=phi0,
        psi0_deg=psi0,
        grating_type=grating_type,
        Lambda_um=Lambda_um,
        thickness_total_um=thickness_total_um,
        duty_cycle=duty_cycle,
        shift=shift,
        number_of_layers=number_of_layers,
        layer_thickness_m=layer_thickness,
        n1=n1, n3=n3, ng=ng, nr=nr,
        measurement=0,
        diffraction_efficiencies_c=1,
        studying_order=0,
        device=device, dtype=dtype, cdtype=cdtype,
        eps=torch.tensor(1e-12, device=device, dtype=dtype),
    )

    core_out = rcwa_procedure_fn(ctx)
    R_s = core_out.R_s.to(cdtype)
    R_p = core_out.R_p.to(cdtype)

    mid_order = (number_of_orders + 1) // 2
    Rs_0 = R_s[mid_order - 1]
    Rp_0 = (1j / n1) * R_p[mid_order - 1]
    return Rs_0, Rp_0


# -------------------- 扫描并汇总结果 --------------------

def compute_section2_results(para, rcwa_procedure_fn=None):
    """
    扫描 λ 与角度，计算 0 级 Jones/Mueller 与 r_sp/r_pp/r_ss/r_ps 及 Psi/Delta。
    输入:
      - para: build_section1_params 返回的对象（或等价结构）
    输出:
      - SimpleNamespace，字段:
          Muller_list: [L_lambda, 16, L_theta] (real, para.dtype)
          r_sp_list, r_pp_list, r_ss_list, r_ps_list: [L_lambda, 1, L_theta] (complex, para.cdtype)
          Psi_Delta_list: [L_lambda, 6, L_theta] (real, para.dtype)
    """
    device = para.device if hasattr(para, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = para.dtype  if hasattr(para, "dtype")  else torch.float32
    cdtype = para.cdtype if hasattr(para, "cdtype") else (torch.complex64 if dtype == torch.float32 else torch.complex128)

    # 小工具：把 tuple/list/tensor 统一成 1D tensor
    def _as_1d_tensor(x, device, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype).reshape(-1)
        return torch.as_tensor(x, device=device, dtype=dtype).reshape(-1)

    # 读取并标准化扫描列表
    theta_list = _as_1d_tensor(
        getattr(para.incident, "theta0_list_deg", getattr(para.incident, "theta0_list", None)),
        device, dtype
    )
    phi_list = _as_1d_tensor(
        getattr(para.incident, "phi0_list_deg", getattr(para.incident, "phi0_list", None)),
        device, dtype
    )
    lambda_list = _as_1d_tensor(
        getattr(para.incident, "lambda_list_um", getattr(para.incident, "lambda_list", None)),
        device, dtype
    )

    L_theta  = int(theta_list.numel())
    L_lambda = int(lambda_list.numel())

    # 取出材料色散列表（随 λ 变化的复折射率）
    n1_list = para.grating.n1_list.reshape(-1).to(device=device, dtype=cdtype)
    n3_list = para.grating.n3_list.reshape(-1).to(device=device, dtype=cdtype)
    ng_list = para.grating.ng_list.reshape(-1).to(device=device, dtype=cdtype)
    nr_list = para.grating.nr_list.reshape(-1).to(device=device, dtype=cdtype)
    # 简单一致性检查
    if not (n1_list.numel() == n3_list.numel() == ng_list.numel() == nr_list.numel() == L_lambda):
        raise ValueError("n1/n3/ng/nr 列表长度必须与 lambda_list 一致。")

    # 预分配输出
    Muller_list    = torch.zeros(L_lambda, 16, L_theta, device=device, dtype=dtype)
    r_sp_list      = torch.zeros(L_lambda, 1,  L_theta, device=device, dtype=cdtype)
    r_pp_list      = torch.zeros(L_lambda, 1,  L_theta, device=device, dtype=cdtype)
    r_ss_list      = torch.zeros(L_lambda, 1,  L_theta, device=device, dtype=cdtype)
    r_ps_list      = torch.zeros(L_lambda, 1,  L_theta, device=device, dtype=cdtype)
    Psi_Delta_list = torch.zeros(L_lambda, 6,  L_theta, device=device, dtype=dtype)

    # 可选：允许自定义 rcwa_procedure；默认用文件内定义的
    if rcwa_procedure_fn is None:
        rcwa_procedure_fn = rcwa_procedure_torch  # noqa: F821 - 本文件中应已有定义

    # 扫描
    for kk1 in range(L_theta):
        theta0 = theta_list[kk1]
        # 如果 phi 列表与 theta 一样长，逐一对应；否则用第一个
        phi0 = phi_list[kk1] if phi_list.numel() == L_theta else phi_list[0]

        # 写入 para 的当前角度（保持 tensor，内部用 torch 计算）
        para.incident.theta0 = theta0
        para.incident.phi0   = phi0

        for kk2 in range(L_lambda):
            lam_um = lambda_list[kk2]
            # 避免使用关键字属性名 lambda
            para.incident.lambda_um = lam_um

            # 当前波长的材料参数
            para.grating.n1 = n1_list[kk2]
            para.grating.n3 = n3_list[kk2]
            para.grating.ng = ng_list[kk2]
            para.grating.nr = nr_list[kk2]

            # TM 入射 (psi0 = 0)
            para.incident.psi0 = torch.as_tensor(0.0, device=device, dtype=dtype)
            Rsp0, Rpp0 = simu_rcwa1d_main_torch(para, rcwa_procedure_fn=rcwa_procedure_fn)  # noqa: F821

            # TE 入射 (psi0 = 90)
            para.incident.psi0 = torch.as_tensor(90.0, device=device, dtype=dtype)
            Rss0, Rps0 = simu_rcwa1d_main_torch(para, rcwa_procedure_fn=rcwa_procedure_fn)

            # Jones / Muller（注意：我们的 jones_muller_torch 已与 MATLAB 完全对齐）
            _, Muller_vec = jones_muller_torch(Rsp0, Rpp0, Rss0, Rps0, para.grating.n1)  # noqa: F821

            # Psi/Delta
            r_pp_ss = Rpp0 / Rss0
            r_ps_ss = Rps0 / Rss0
            r_sp_ss = Rsp0 / Rss0

            Psi_pp   = torch.rad2deg(torch.atan(torch.abs(r_pp_ss)))
            Delta_pp = torch.rad2deg(torch.angle(r_pp_ss))
            Psi_ps   = torch.rad2deg(torch.atan(torch.abs(r_ps_ss)))
            Delta_ps = torch.rad2deg(torch.angle(r_ps_ss))
            Psi_sp   = torch.rad2deg(torch.atan(torch.abs(r_sp_ss)))
            Delta_sp = torch.rad2deg(torch.angle(r_sp_ss))

            # 填充输出（与 MATLAB 维度一致）
            Muller_list[kk2, :, kk1]    = Muller_vec.real.to(dtype)
            r_sp_list[kk2, 0, kk1]      = Rsp0
            r_pp_list[kk2, 0, kk1]      = Rpp0
            r_ss_list[kk2, 0, kk1]      = Rss0
            r_ps_list[kk2, 0, kk1]      = Rps0
            Psi_Delta_list[kk2, :, kk1] = torch.stack([Psi_pp, Delta_pp, Psi_ps, Delta_ps, Psi_sp, Delta_sp]).to(dtype)

    return SN(
        Muller_list=Muller_list,
        r_sp_list=r_sp_list, r_pp_list=r_pp_list, r_ss_list=r_ss_list, r_ps_list=r_ps_list,
        Psi_Delta_list=Psi_Delta_list,
    )