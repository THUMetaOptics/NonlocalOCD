from __future__ import annotations

from types import SimpleNamespace as SN
from typing import List, Tuple

import torch


def _mrdivide(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """MATLAB A / B = (B' \\ A')' using conjugate-transpose for complex matrices."""
    return torch.linalg.solve(B.mH, A.mH).mH


def toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    if c.dim() != 1 or r.dim() != 1:
        raise ValueError('toeplitz expects 1D inputs')
    c = c.reshape(-1)
    r = r.reshape(-1)
    if c.numel() > 0 and r.numel() > 0 and not torch.allclose(c[:1], r[:1]):
        r = r.clone()
        r[0] = c[0]
    p = r.numel()
    m = c.numel()
    x = torch.cat([r[1:].flip(0), c], dim=0)
    ij = torch.arange(m, device=c.device).reshape(-1, 1) + torch.arange(p - 1, -1, -1, device=c.device).reshape(1, -1)
    return x[ij]


def cotd(x_deg: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tan(torch.deg2rad(x_deg))


def trapezoidal_grating_setup(para: SN) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = para.grating.Lambda.device
    rdtype = para.grating.Lambda.dtype

    Lambda = para.grating.Lambda.reshape(())
    thickness_total = para.grating.thickness_total.reshape(())
    TCD = para.grating.TCD.reshape(())
    angle_trap = para.grating.angle_trap.reshape(())
    number_of_layers = int(para.grating.number_of_layers)

    a_1 = (Lambda - TCD) / (2.0 * Lambda)
    a_2 = a_1
    BCD = TCD + 2.0 * thickness_total * cotd(angle_trap)
    b = BCD / Lambda

    if bool((a_1 + a_2 > 1).item()):
        raise ValueError('Trapezoidal shape parameter setting error!!!')

    d = thickness_total
    y_1 = torch.zeros((), device=device, dtype=rdtype)
    x_1 = (Lambda - b * Lambda) / 2.0
    y_2 = torch.zeros((), device=device, dtype=rdtype)
    x_2 = (Lambda + b * Lambda) / 2.0
    y_3 = d
    x_3 = a_1 * Lambda
    y_4 = d
    x_4 = Lambda - a_2 * Lambda

    rozdeleni = torch.linspace(0.0, 1.0, number_of_layers + 1, device=device, dtype=rdtype) * d
    nova_y = 0.5 * (rozdeleni[:-1] + rozdeleni[1:])

    zero = torch.zeros((), device=device, dtype=rdtype)
    if torch.isclose(x_3 - x_1, zero):
        pomocna_1 = x_1 * torch.ones(number_of_layers, device=device, dtype=rdtype)
    else:
        pomocna_1 = (nova_y - (y_1 * x_3 - y_3 * x_1) / (x_3 - x_1)) * ((x_1 - x_3) / (y_1 - y_3))

    if torch.isclose(x_4 - x_2, zero):
        pomocna_2 = x_2 * torch.ones(number_of_layers, device=device, dtype=rdtype)
    else:
        pomocna_2 = (nova_y - (y_2 * x_4 - y_4 * x_2) / (x_4 - x_2)) * ((x_2 - x_4) / (y_2 - y_4))

    duty_cycle = ((pomocna_2 - pomocna_1) / Lambda).flip(0)
    layer_thickness = (d * 1e-6 / number_of_layers) * torch.ones(number_of_layers, device=device, dtype=rdtype)
    shift = ((pomocna_1 + pomocna_2) / (2.0 * Lambda)).flip(0)
    return layer_thickness, duty_cycle, shift


def eps_fourier_series(epsg: torch.Tensor, epsr: torch.Tensor, duty: torch.Tensor, M: int, shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = epsg.device
    rdtype = epsg.real.dtype
    cdtype = epsg.dtype

    m = torch.arange(1, M + 1, device=device, dtype=rdtype)
    Sinc = torch.sin(torch.pi * duty * m) / (torch.pi * m)

    epsG = (1.0 - duty) * epsg + duty * epsr
    v_m = (epsr - epsg) * torch.flip(Sinc, dims=(0,))
    v_0 = epsG.unsqueeze(0)
    v_p = (epsr - epsg) * Sinc
    phase = torch.exp(-1j * 2.0 * torch.pi * shift * torch.arange(-M, M + 1, device=device, dtype=rdtype)).to(cdtype)
    v = torch.cat([v_m, v_0, v_p], dim=0).to(cdtype) * phase

    inv_epsr = 1.0 / epsr
    inv_epsg = 1.0 / epsg
    i_epsG = (1.0 - duty) * inv_epsg + duty * inv_epsr
    i_vm = (inv_epsr - inv_epsg) * torch.flip(Sinc, dims=(0,))
    i_v0 = i_epsG.unsqueeze(0)
    i_vp = (inv_epsr - inv_epsg) * Sinc
    i_v = torch.cat([i_vm, i_v0, i_vp], dim=0).to(cdtype) * phase
    return v, i_v


def _kz_branch(k0: torch.Tensor, n_med: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lhs = kx ** 2 + ky ** 2
    rhs = (k0 * n_med) ** 2
    cond = torch.real(lhs) < torch.real(rhs)

    kz = torch.empty_like(kx)
    kz[cond] = torch.sqrt(rhs - lhs)[cond]
    kz[~cond] = (-1j * torch.sqrt(lhs - rhs))[~cond]

    kz_n = kz.clone()
    is_real = torch.isclose(kz.imag, torch.zeros_like(kz.imag), atol=1e-14, rtol=0.0)
    kz_n = torch.where(is_real, -kz, kz_n)
    return kz, kz_n


def S_matrix_initialization(phi_i: torch.Tensor, k_1_z: torch.Tensor, k_3_z: torch.Tensor, k_0: torch.Tensor, n1: torch.Tensor, n3: torch.Tensor, I: torch.Tensor):
    F_c = torch.diag(torch.cos(phi_i).to(dtype=I.dtype))
    F_s = torch.diag(torch.sin(phi_i).to(dtype=I.dtype))

    k_1_z_k_0 = k_1_z / k_0
    Y1 = torch.diag(k_1_z_k_0)
    Z1 = torch.diag(k_1_z_k_0 / (n1 ** 2))

    k_3_z_k_0 = k_3_z / k_0
    Y3 = torch.diag(k_3_z_k_0)
    Z3 = torch.diag(k_3_z_k_0 / (n3 ** 2))
    return F_c, F_s, Y1, Z1, Y3, Z3


def S_Li_conical_final(number_of_orders: int, number_of_layers: int, I: torch.Tensor, zero: torch.Tensor, Y1: torch.Tensor, Z1: torch.Tensor, Y3: torch.Tensor, Z3: torch.Tensor, psi: torch.Tensor, theta: torch.Tensor, n1: torch.Tensor, k_0: torch.Tensor, Q1_list: List[torch.Tensor], Q2_list: List[torch.Tensor], layer_thickness: torch.Tensor, Vss_list: List[torch.Tensor], Vsp_list: List[torch.Tensor], Wss_list: List[torch.Tensor], Wsp_list: List[torch.Tensor], Wpp_list: List[torch.Tensor], Vpp_list: List[torch.Tensor], Wps_list: List[torch.Tensor], Vps_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cdtype = I.dtype
    device = I.device
    M2 = 2 * number_of_orders
    M4 = 4 * number_of_orders

    W_3_11 = torch.cat([torch.cat([I, zero], dim=1), torch.cat([-1j * Y1, zero], dim=1)], dim=0)
    W_3_12 = torch.cat([torch.cat([torch.sin(psi) * I, zero], dim=1), torch.cat([1j * n1 * torch.cos(theta) * torch.sin(psi) * I, zero], dim=1)], dim=0)
    W_3_21 = torch.cat([torch.cat([zero, I], dim=1), torch.cat([zero, -1j * Z1], dim=1)], dim=0)
    W_3_22 = torch.cat([torch.cat([-1j * n1 * torch.cos(psi) * I, zero], dim=1), torch.cat([torch.cos(psi) * torch.cos(theta) * I, zero], dim=1)], dim=0)

    W_2_11 = []
    W_2_12 = []
    W_2_21 = []
    W_2_22 = []
    for i in range(number_of_layers):
        W_2_11.append(torch.cat([torch.cat([Vss_list[i], Vsp_list[i]], dim=1), torch.cat([-Wss_list[i], -Wsp_list[i]], dim=1)], dim=0))
        W_2_12.append(torch.cat([torch.cat([Vss_list[i], Vsp_list[i]], dim=1), torch.cat([Wss_list[i], Wsp_list[i]], dim=1)], dim=0))
        W_2_21.append(torch.cat([torch.cat([-Wps_list[i], -Wpp_list[i]], dim=1), torch.cat([Vps_list[i], Vpp_list[i]], dim=1)], dim=0))
        W_2_22.append(torch.cat([torch.cat([Wps_list[i], Wpp_list[i]], dim=1), torch.cat([Vps_list[i], Vpp_list[i]], dim=1)], dim=0))

    W_1_11 = torch.cat([torch.cat([zero, zero], dim=1), torch.cat([zero, zero], dim=1)], dim=0)
    W_1_12 = torch.cat([torch.cat([I, zero], dim=1), torch.cat([1j * Y3, zero], dim=1)], dim=0)
    W_1_21 = torch.cat([torch.cat([zero, zero], dim=1), torch.cat([zero, zero], dim=1)], dim=0)
    W_1_22 = torch.cat([torch.cat([zero, I], dim=1), torch.cat([zero, 1j * Z3], dim=1)], dim=0)

    s_2 = torch.linalg.solve(
        torch.cat([torch.cat([W_3_11, -W_2_12[0]], dim=1), torch.cat([W_3_21, -W_2_22[0]], dim=1)], dim=0),
        torch.cat([torch.cat([W_2_11[0], -W_3_12], dim=1), torch.cat([W_2_21[0], -W_3_22], dim=1)], dim=0),
    )

    s_1_list = []
    for i in range(1, number_of_layers):
        upper = number_of_layers - i - 1
        lower = number_of_layers - i
        s_1 = torch.linalg.solve(
            torch.cat([torch.cat([W_2_11[upper], -W_2_12[lower]], dim=1), torch.cat([W_2_21[upper], -W_2_22[lower]], dim=1)], dim=0),
            torch.cat([torch.cat([W_2_11[lower], -W_2_12[upper]], dim=1), torch.cat([W_2_21[lower], -W_2_22[upper]], dim=1)], dim=0),
        )
        s_1_list.append(s_1)

    s_0 = torch.linalg.solve(
        torch.cat([torch.cat([W_2_11[-1], -W_1_12], dim=1), torch.cat([W_2_21[-1], -W_1_22], dim=1)], dim=0),
        torch.cat([torch.cat([W_1_11, -W_2_12[-1]], dim=1), torch.cat([W_1_21, -W_2_22[-1]], dim=1)], dim=0),
    )

    X_1 = []
    X_2 = []
    for i in range(number_of_layers):
        idx = number_of_layers - 1 - i
        X_1.append(torch.diag(torch.exp(-k_0 * torch.diagonal(Q1_list[idx]) * layer_thickness[idx])))
        X_2.append(torch.diag(torch.exp(-k_0 * torch.diagonal(Q2_list[idx]) * layer_thickness[idx])))

    a_ud = [None] * (number_of_layers + 1)
    a_dd = [None] * (number_of_layers + 1)
    a_ud[0] = s_0[:M2, M2:M4]
    a_dd[0] = s_0[M2:M4, M2:M4]

    new_I = torch.block_diag(I, I)

    for i in range(number_of_layers):
        if i == number_of_layers - 1:
            first_m = torch.block_diag(I, I, X_1[i], X_2[i])
            second_m = torch.block_diag(X_1[i], X_2[i], I, I)
            s_2_l = first_m @ s_2 @ second_m
            b_uu = s_2_l[:M2, :M2]
            b_ud = s_2_l[:M2, M2:M4]
            b_du = s_2_l[M2:M4, :M2]
            b_dd = s_2_l[M2:M4, M2:M4]
        else:
            first_m = torch.block_diag(I, I, X_1[i], X_2[i])
            second_m = torch.block_diag(X_1[i], X_2[i], I, I)
            left_diag = torch.diagonal(first_m).reshape(-1, 1)
            # MATLAB: conj((diag(second_m))')\n            # Since ' is conjugate-transpose in MATLAB, the outer conj cancels that
            # conjugation. The net result is a plain row transpose with NO conjugation.
            right_diag = torch.diagonal(second_m).reshape(1, -1)
            s_1_l = (left_diag @ right_diag) * s_1_list[i]
            b_uu = s_1_l[:M2, :M2]
            b_ud = s_1_l[:M2, M2:M4]
            b_du = s_1_l[M2:M4, :M2]
            b_dd = s_1_l[M2:M4, M2:M4]

        denom = new_I - b_du @ a_ud[i]
        a_ud[i + 1] = b_ud + b_uu @ _mrdivide(a_ud[i], denom) @ b_dd
        a_dd[i + 1] = _mrdivide(a_dd[i], denom) @ b_dd

    T_dd = a_dd[-1]
    R_ud = a_ud[-1]
    d_n1 = torch.zeros((2 * number_of_orders, 1), dtype=cdtype, device=device)
    d_n1[(number_of_orders - 1) // 2, 0] = 1.0
    R = R_ud @ d_n1
    T = T_dd @ d_n1
    R_s = R[:number_of_orders, 0]
    R_p = R[number_of_orders:, 0]
    T_s = T[:number_of_orders, 0]
    T_p = T[number_of_orders:, 0]
    return R_s, R_p, T_s, T_p


def rcwa_procedure(para: SN, number_of_orders: int, change_matrix_base: int, polarization: int, faktorization: int, matrix_algorithm: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if polarization != 0:
        raise NotImplementedError('This streamlined implementation only keeps conical diffraction (polarization=0).')
    if matrix_algorithm != 1:
        raise NotImplementedError('This streamlined implementation only keeps matrix_algorithm=1.')
    if change_matrix_base != 1:
        raise NotImplementedError('This streamlined implementation only keeps change_matrix_base=1.')

    n1 = para.grating.n1
    n3 = para.grating.n3
    device = n1.device
    cdtype = n1.dtype
    rdtype = n1.real.dtype

    theta = torch.as_tensor(para.incident.theta0, device=device, dtype=rdtype).reshape(()) * torch.pi / 180.0
    phi = torch.as_tensor(para.incident.phi0, device=device, dtype=rdtype).reshape(()) * torch.pi / 180.0
    psi = torch.as_tensor(para.incident.psi0, device=device, dtype=rdtype).reshape(()) * torch.pi / 180.0
    if torch.isclose(theta, torch.zeros((), device=device, dtype=rdtype)):
        theta = torch.as_tensor(1e-20, device=device, dtype=rdtype)

    lambda_um = getattr(para.incident, 'lambda_', None)
    if lambda_um is None:
        lambda_um = getattr(para.incident, 'lambda')
    lambda_um = torch.as_tensor(lambda_um, device=device, dtype=rdtype).reshape(())

    k_0 = (2.0 * torch.pi / (lambda_um * 1e-6)).to(cdtype)
    Lambda = para.grating.Lambda.reshape(())
    K = (2.0 * torch.pi / (Lambda * 1e-6)).to(cdtype)

    order_max = (number_of_orders - 1) // 2
    n = torch.arange(-order_max, order_max + 1, device=device, dtype=rdtype).to(cdtype)
    I = torch.eye(number_of_orders, dtype=cdtype, device=device)
    zero = torch.zeros((number_of_orders, number_of_orders), dtype=cdtype, device=device)

    k_x = k_0 * n1 * torch.sin(theta) * torch.cos(phi) + n * K
    K_x = torch.diag(k_x / k_0)
    K_x_2 = K_x @ K_x

    k_y = k_0 * n1 * torch.sin(theta) * torch.sin(phi)
    k_y_k_0 = k_y / k_0
    k_y_2_I = (k_y ** 2) * I / (k_0 * k_0)

    if torch.isclose(torch.abs(k_y), torch.zeros((), device=device, dtype=rdtype)):
        phi_i = torch.zeros(number_of_orders, device=device, dtype=rdtype)
    else:
        phi_i = torch.atan((k_y / k_x).real)

    k_1_z, k_1_z_n = _kz_branch(k_0, n1, k_x, k_y)
    k_3_z, k_3_z_n = _kz_branch(k_0, n3, k_x, k_y)

    F_c, F_s, Y1, Z1, Y3, Z3 = S_matrix_initialization(phi_i, k_1_z, k_3_z, k_0, n1, n3, I)

    number_of_layers = int(para.grating.number_of_layers)
    layer_thickness = para.grating.layer_thickness.to(device=device, dtype=rdtype)
    duty_cycle = para.grating.duty_cycle.to(device=device, dtype=rdtype)
    shift = para.grating.shift.to(device=device, dtype=rdtype)

    epsg = para.grating.ng ** 2
    epsr = para.grating.nr ** 2
    M = number_of_orders - 1

    Q1_list: List[torch.Tensor] = []
    Q2_list: List[torch.Tensor] = []
    Vss_list: List[torch.Tensor] = []
    Wss_list: List[torch.Tensor] = []
    Vsp_list: List[torch.Tensor] = []
    Wsp_list: List[torch.Tensor] = []
    Wpp_list: List[torch.Tensor] = []
    Vpp_list: List[torch.Tensor] = []
    Wps_list: List[torch.Tensor] = []
    Vps_list: List[torch.Tensor] = []

    for l in range(number_of_layers):
        v, i_v = eps_fourier_series(epsg, epsr, duty_cycle[l], M, shift[l])
        E_l = toeplitz(torch.flip(v[:number_of_orders], dims=(0,)), v[number_of_orders - 1: 2 * number_of_orders - 1])
        A_l = toeplitz(torch.flip(i_v[:number_of_orders], dims=(0,)), i_v[number_of_orders - 1: 2 * number_of_orders - 1])

        A = K_x_2 - E_l
        eigenvalue_equation_1 = k_y_2_I + K_x_2 - E_l
        eval1, W1 = torch.linalg.eig(eigenvalue_equation_1)
        Q1 = torch.diag(torch.sqrt(eval1))

        inv_E_l = torch.linalg.inv(E_l)
        B = K_x @ inv_E_l @ K_x - I
        if faktorization == 1:
            eigenvalue_equation_2 = k_y_2_I + _mrdivide(B, A_l)
        elif faktorization == 2:
            eigenvalue_equation_2 = k_y_2_I + B @ E_l
        else:
            raise ValueError('faktorization must be 1 or 2')

        eval2, W2 = torch.linalg.eig(eigenvalue_equation_2)
        Q2 = torch.diag(torch.sqrt(eval2))

        inv_A = torch.linalg.inv(A)
        inv_B = torch.linalg.inv(B)
        V11 = inv_A @ W1 @ Q1
        V12 = k_y_k_0 * (inv_A @ K_x @ W2)
        V21 = k_y_k_0 * (inv_B @ K_x @ inv_E_l @ W1)
        V22 = inv_B @ W2 @ Q2

        Vss = F_c @ V11
        Wss = F_c @ W1 + F_s @ V21
        Vsp = F_c @ V12 - F_s @ W2
        Wsp = F_s @ V22
        Wpp = F_c @ V22
        Vpp = F_c @ W2 + F_s @ V12
        Wps = F_c @ V21 - F_s @ W1
        Vps = F_s @ V11

        Q1_list.append(Q1)
        Q2_list.append(Q2)
        Vss_list.append(Vss)
        Wss_list.append(Wss)
        Vsp_list.append(Vsp)
        Wsp_list.append(Wsp)
        Wpp_list.append(Wpp)
        Vpp_list.append(Vpp)
        Wps_list.append(Wps)
        Vps_list.append(Vps)

    R_s, R_p, T_s, T_p = S_Li_conical_final(
        number_of_orders=number_of_orders,
        number_of_layers=number_of_layers,
        I=I,
        zero=zero,
        Y1=Y1,
        Z1=Z1,
        Y3=Y3,
        Z3=Z3,
        psi=psi.to(cdtype),
        theta=theta.to(cdtype),
        n1=n1,
        k_0=k_0,
        Q1_list=Q1_list,
        Q2_list=Q2_list,
        layer_thickness=layer_thickness.to(cdtype),
        Vss_list=Vss_list,
        Vsp_list=Vsp_list,
        Wss_list=Wss_list,
        Wsp_list=Wsp_list,
        Wpp_list=Wpp_list,
        Vpp_list=Vpp_list,
        Wps_list=Wps_list,
        Vps_list=Vps_list,
    )
    return R_s, R_p
