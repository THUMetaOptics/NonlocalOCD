import torch
import math


def R(theta):
    """
    根据输入角度 theta（度）计算并返回一个 4x4 的矩阵 R。

    参数:
        theta (torch.Tensor): 输入角度，形状为 [..., 1] 或 [...],
                             单位为度。
    返回:
        torch.Tensor: 形状为 [..., 4, 4] 的 R 矩阵。
    """
    # 如果最后一维是 1，则压缩掉
    if theta.shape[-1] == 1:
        theta = theta.squeeze(-1)  # [N, 241] (例如)

    # 将角度从度转换为弧度
    theta_rad = theta * (math.pi / 180)

    # 计算 cos(2*theta) 和 sin(2*theta)
    cos_2theta = torch.cos(2 * theta_rad)
    sin_2theta = torch.sin(2 * theta_rad)

    # 为了方便构建矩阵，先准备好一些零和一
    ones = torch.ones_like(theta_rad)
    zeros = torch.zeros_like(theta_rad)

    # 对应 Matlab:
    # [1,        0,         0,        0
    #  0,   cos(2θ),   sin(2θ),       0
    #  0,  -sin(2θ),   cos(2θ),       0
    #  0,        0,         0,       1 ]
    row1 = torch.stack([ones, zeros, zeros, zeros], dim=-1)
    row2 = torch.stack([zeros, cos_2theta, sin_2theta, zeros], dim=-1)
    row3 = torch.stack([zeros, -sin_2theta, cos_2theta, zeros], dim=-1)
    row4 = torch.stack([zeros, zeros, zeros, ones], dim=-1)

    # 拼接成最终的 [..., 4, 4] 形状
    R_mat = torch.stack([row1, row2, row3, row4], dim=-2)
    return R_mat


def M_pol(theta_lp):
    """
    根据输入角度 theta_lp（度）计算并返回一个 4x4 的矩阵 Mpol。

    参数:
        theta_lp (torch.Tensor): 输入角度，形状为 [..., 1] 或 [...],
                                 单位为度。
    返回:
        torch.Tensor: 形状为 [..., 4, 4] 的 Mpol 矩阵。
    """
    # 如果最后一维是 1，则压缩
    if theta_lp.shape[-1] == 1:
        theta_lp = theta_lp.squeeze(-1)

    # 转为弧度
    theta_rad = theta_lp * (math.pi / 180)
    cos_2theta = torch.cos(2 * theta_rad)
    sin_2theta = torch.sin(2 * theta_rad)

    # 预防数值不稳定
    cos_2theta = torch.clamp(cos_2theta, min=-1.0, max=1.0)
    sin_2theta = torch.clamp(sin_2theta, min=-1.0, max=1.0)

    # 准备 zeros & ones
    ones = torch.ones_like(theta_rad)
    zeros = torch.zeros_like(theta_rad)

    # 对应 Matlab:
    # [ 1,  cos(2θ),              sin(2θ),              0
    #   cos(2θ),  cos(2θ)^2,      sin(2θ)*cos(2θ),       0
    #   sin(2θ),  sin(2θ)*cos(2θ), sin(2θ)^2,            0
    #   0,        0,              0,                    0 ] / 2
    row1 = torch.stack([ones, cos_2theta, sin_2theta, zeros], dim=-1)
    row2 = torch.stack([cos_2theta, cos_2theta ** 2, sin_2theta * cos_2theta, zeros], dim=-1)
    row3 = torch.stack([sin_2theta, sin_2theta * cos_2theta, sin_2theta ** 2, zeros], dim=-1)
    row4 = torch.stack([zeros, zeros, zeros, zeros], dim=-1)

    # 堆叠在第 -2 维，得到 [..., 4, 4]
    Mpol_mat = torch.stack([row1, row2, row3, row4], dim=-2) / 2.0

    return Mpol_mat


def M_wp(phi_wp):
    """
    根据输入角度 phi_wp（度）计算并返回一个 4x4 的矩阵 Mwp。

    参数:
        phi_wp (torch.Tensor): 输入角度，形状为 [..., 1] 或 [...],
                              单位为度。
    返回:
        torch.Tensor: 形状为 [..., 4, 4] 的 Mwp 矩阵。
    """
    # 如果最后一维是 1，则压缩
    if phi_wp.shape[-1] == 1:
        phi_wp = phi_wp.squeeze(-1)

    # 转为弧度
    phi_rad = phi_wp * (math.pi / 180)

    # 计算 cos(phi) 和 sin(phi)
    c = torch.cos(phi_rad)
    s = torch.sin(phi_rad)

    # 准备 ones & zeros
    ones = torch.ones_like(phi_rad)
    zeros = torch.zeros_like(phi_rad)

    # 对应 Matlab:
    # [1, 0,      0,      0
    #  0, 1,      0,      0
    #  0, 0,  cos(phi), -sin(phi)
    #  0, 0,  sin(phi),  cos(phi)]
    row1 = torch.stack([ones, zeros, zeros, zeros], dim=-1)
    row2 = torch.stack([zeros, ones, zeros, zeros], dim=-1)
    row3 = torch.stack([zeros, zeros, c, -s], dim=-1)
    row4 = torch.stack([zeros, zeros, s, c], dim=-1)

    Mwp_mat = torch.stack([row1, row2, row3, row4], dim=-2)
    return Mwp_mat


# 示例使用
if __name__ == "__main__":
    # 示例 1：标量输入
    # theta_scalar = torch.tensor([0.5])  # 例如，theta_lp = 0.5 radians
    # print(theta_scalar.shape)
    # Mpol_scalar = M_pol(theta_scalar)
    # print("标量输入的输出 Mpol:")
    # # print(Mpol_scalar)
    # print("输出尺寸:", Mpol_scalar.shape)  # 应该是 [4, 4]
    #
    # # 示例 2：张量输入
    # theta_tensor = torch.tensor([0.0, torch.pi / 4, torch.pi / 2])  # 例如，三个不同的角度
    # Mpol_tensor = M_pol(theta_tensor)
    # print("\n张量输入的输出 Mpol:")
    # # print(Mpol_tensor)
    # print("输出尺寸:", Mpol_tensor.shape)  # 应该是 [3, 4, 4]
    #
    # # 示例 3：张量输入
    # theta_tensor = torch.randn(10, 121, 1)
    # Mpol_tensor = M_pol(theta_tensor)
    # print("\n张量输入的输出 Mpol:")
    # # print(Mpol_tensor)
    # print("输出尺寸:", Mpol_tensor.shape)  # 应该是 [3, 4, 4]

    theta = torch.ones([10, 121, 1]) * 90
    Ms = M_pol(theta)
    print(Ms.shape)

    theta_R = torch.ones(1, 1) * 90
    cahce = R(theta_R * -1) @ M_wp(theta_R) @ R(theta_R)
    print(cahce.shape)
    Ms = Ms@cahce
    print(Ms.shape)

    Ms = Ms.unsqueeze(1).unsqueeze(1)
    print(Ms.shape)



