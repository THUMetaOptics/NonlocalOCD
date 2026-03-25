from __future__ import annotations

import torch


def cal_Jones_and_Muller_matrix_useTMTE_0order(Rs0_TM, Rp0_TM, Rs0_TE, Rp0_TE, n1):
    cd = Rs0_TM.dtype
    rd = Rs0_TM.real.dtype
    dev = Rs0_TM.device

    J22 = Rs0_TE  # r_ss
    J12 = Rp0_TE  # r_ps
    J21 = Rs0_TM  # r_sp
    J11 = Rp0_TM  # r_pp
    Jones_matrix = torch.stack([
        torch.stack([J11, J12]),
        torch.stack([J21, J22]),
    ], dim=0).to(dtype=cd, device=dev)

    U = torch.tensor(
        [[1, 0, 0, 1],
         [1, 0, 0, -1],
         [0, 1, 1, 0],
         [0, 1j, -1j, 0]],
        dtype=cd,
        device=dev,
    )

    Muller_matrix = U @ torch.kron(Jones_matrix, Jones_matrix.conj()) @ torch.linalg.inv(U)
    Muller_matrix = Muller_matrix.transpose(0, 1).clone()

    Muller_matrix[0, 2] = -Muller_matrix[0, 2]
    Muller_matrix[1, 2] = -Muller_matrix[1, 2]
    Muller_matrix[2, 0] = -Muller_matrix[2, 0]
    Muller_matrix[2, 1] = -Muller_matrix[2, 1]
    Muller_matrix[2, 3] = -Muller_matrix[2, 3]
    Muller_matrix[3, 2] = -Muller_matrix[3, 2]

    Muller_vector = Muller_matrix.transpose(0, 1).reshape(-1)
    return Jones_matrix, Muller_vector.real.to(rd)
