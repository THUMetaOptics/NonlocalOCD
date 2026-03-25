import math
from typing import Optional, Union, Tuple
from types import SimpleNamespace as SN
import torch
import torch.nn as nn
from section1_params import build_section1_params
from section2_scan import compute_section2_results


def compute_Iout_vectorized(ms, Sencoder, Spec, Mdecoder, *, eps=1e-12):
    if not isinstance(ms, torch.Tensor):
        raise TypeError("ms must be a torch.Tensor")
    ref_dev, ref_dtype = ms.device, ms.dtype
    Sencoder = torch.as_tensor(Sencoder, device=ref_dev, dtype=ref_dtype)
    Spec     = torch.as_tensor(Spec,     device=ref_dev, dtype=ref_dtype)
    Mdecoder = torch.as_tensor(Mdecoder, device=ref_dev, dtype=ref_dtype)
    ms = ms.unsqueeze(0).unsqueeze(1).unsqueeze(1)
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


class RCWAForward(nn.Module):

    def __init__(
        self,
        *,
        air_path: str,
        si_path: str,
        # 入射角设置
        theta_deg: float = 55.0,
        phi_deg: float = 0.0,
        # 波长扫描区间（μm）
        lambda_um_range: Tuple[float, float, float] = (0.480, 0.720, 0.002),
        # RCWA 阶数
        number_of_orders: int = 41,
        # 设备&精度
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,  # None → 自动：cuda→float32, cpu→float64
    ):
        super().__init__()
        self.device = torch.device(device)
        if dtype is None:
            self.rdtype = torch.float32 if self.device.type == "cuda" else torch.float64
        else:
            self.rdtype = dtype
        self.cdtype = torch.complex64 if self.rdtype == torch.float32 else torch.complex128

        self.air_path = air_path
        self.si_path = si_path
        self.theta_deg = float(theta_deg)
        self.phi_deg = float(phi_deg)
        self.number_of_orders = int(number_of_orders)
        self.lambda_um_start, self.lambda_um_end, self.lambda_um_step = lambda_um_range
        self._P_base = build_section1_params(
            grating_lambda=torch.tensor(0.8, device=self.device, dtype=self.rdtype),
            grating_duty_cycle=torch.tensor(0.5, device=self.device, dtype=self.rdtype),
            grating_thickness_total=torch.tensor(0.5, device=self.device, dtype=self.rdtype),
            number_of_orders=self.number_of_orders,
            lambda_um_start=self.lambda_um_start,
            lambda_um_end=self.lambda_um_end,
            lambda_um_step=self.lambda_um_step,
            theta0_list_deg=(self.theta_deg,),
            phi0_list_deg=(self.phi_deg,),
            n1_lambda_n_k_filename=self.air_path,
            n3_lambda_n_k_filename=self.si_path,
            ng_lambda_n_k_filename=self.air_path,
            nr_lambda_n_k_filename=self.si_path,
            device=self.device,
            dtype=self.rdtype,
        )

    def _as_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.rdtype)
        return torch.tensor(x, device=self.device, dtype=self.rdtype)

    def _make_para_from_base(self, t, La, dc):
        """用已预加载的材料/波长模板，快速构造每个样本的 para（避免反复读盘/插值）。"""
        P = SN()
        P.device = self.device;
        P.dtype = self.rdtype
        P.cdtype = torch.complex64 if self.rdtype == torch.float32 else torch.complex128

        P.basic = SN(number_of_orders=self.number_of_orders)
        P.incident = SN(
            lambda_list_um=self._P_base.incident.lambda_list_um,
            theta0_list_deg=(self.theta_deg,),
            phi0_list_deg=(self.phi_deg,),
        )

        G0 = self._P_base.grating
        G = SN()
        # 材料频散列表直接复用（与 t/La/dc 无关）
        G.n1_list = G0.n1_list;
        G.n3_list = G0.n3_list
        G.ng_list = G0.ng_list;
        G.nr_list = G0.nr_list
        # 三个标量按当前样本覆盖
        G.Lambda_um = La
        G.thickness_total_um = t
        G.duty_cycle = dc
        G.grating_type = G0.grating_type
        P.grating = G
        return P

    def forward(self, thickness_um, grating_Lambda_um, duty_cycle, *,
                chunk_size: int = None, no_grad: bool = False) -> torch.Tensor:
        """
        输入可为标量或任意可广播形状；输出形状为 [*, L_lambda, 16]
          - * 为广播后的批量形状
          - chunk_size: 将批量分块以控制显存（如 32/64/128）
          - no_grad=True 时禁用 autograd 用于离线数据生成
        """

        # 转 dtype/device（保留梯度）
        def _to(x):
            return x if isinstance(x, torch.Tensor) and x.device == self.device and x.dtype == self.rdtype \
                else torch.as_tensor(x, device=self.device, dtype=self.rdtype)

        t = _to(thickness_um)
        La = _to(grating_Lambda_um)
        dc = _to(duty_cycle)

        # 广播到公共形状
        t, La, dc = torch.broadcast_tensors(t, La, dc)
        batch_shape = t.shape
        N = t.numel()

        # 标量路径（与老接口兼容）
        if N == 1:
            P = self._make_para_from_base(t.reshape(()), La.reshape(()), dc.reshape(()))
            out = compute_section2_results(P)
            return out.Muller_list[:, :, 0]  # [L_lambda, 16]

        # 批量路径：展平 + 分块
        t_f, La_f, dc_f = t.reshape(-1), La.reshape(-1), dc.reshape(-1)
        outs = []

        runner = torch.no_grad() if no_grad else torch.enable_grad()
        with runner:
            if chunk_size is None:
                chunk_size = N  # 全量（注意显存）
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                chunk = []
                for i in range(s, e):
                    P = self._make_para_from_base(t_f[i], La_f[i], dc_f[i])
                    out = compute_section2_results(P)
                    chunk.append(out.Muller_list[:, :, 0])  # [L_lambda, 16]
                outs.append(torch.stack(chunk, dim=0))  # [Bch, L_lambda, 16]

        out_all = torch.cat(outs, dim=0)  # [N, L_lambda, 16]
        return out_all.reshape(*batch_shape, out_all.shape[-2], out_all.shape[-1])


# -------------------------
# 便捷小网络，用于测试 B)
# -------------------------
class TinyDNet(nn.Module):
    def __init__(self, d_min: float = 0.3, d_max: float = 0.8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2, bias=False), nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 3)
        self.d_min, self.d_max = float(d_min), float(d_max)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) == 1:
            pass
        else:
            x = x.unsqueeze(1)
        h = self.backbone(x).flatten(1)
        z = self.head(h)
        s = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
        # 输出 3 个正数，映射到 (d_min, d_max)
        return s * (self.d_max - self.d_min) + self.d_min  # [B,3]


# -------------------------
# 简单自检 & 梯度测试（A & B）
# -------------------------
if __name__ == "__main__":
    # 0) 设备检查
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("CUDA available:", use_cuda)

    # 路径：请改为你的本地 ASCII nk 文件路径
    AIR = r"F:\科研项目\XueXinyuan\2025_I2parameters\光栅\material_lambda_nk\Air_lambda_n_k.mat"
    SI  = r"F:\科研项目\XueXinyuan\2025_I2parameters\光栅\material_lambda_nk\Si_lambda_n_k.mat"

    # 1) 封装模块
    rcwa = RCWAForward(
        air_path=AIR,
        si_path=SI,
        theta_deg=55.0,
        phi_deg=0.0,
        device=device,
        dtype=None,
    )

    # ---------------- A) 直接 nn.Parameter 测试 ----------------
    print("\n[A] nn.Parameter 梯度回传测试")
    thickness = nn.Parameter(torch.tensor(0.472, device=device, dtype=rcwa.rdtype))
    Lambda    = nn.Parameter(torch.tensor(0.800, device=device, dtype=rcwa.rdtype))
    duty      = nn.Parameter(torch.tensor(0.4375, device=device, dtype=rcwa.rdtype))

    ms = rcwa(thickness, Lambda, duty)  # [L_lambda, 16]
    print(ms.shape)
    loss = ms.sum()
    loss.backward()

    print("  thickness.grad:", None if thickness.grad is None else float(thickness.grad))
    print("  Lambda.grad   :", None if Lambda.grad is None else float(Lambda.grad))
    print("  duty.grad     :", None if duty.grad is None else float(duty.grad))

    # ---------------- B) 小网络端到端回传 ----------------
    print("\n[B] TinyDNet → RCWA 端到端回传测试")
    net = TinyDNet().to(device)

    # 构造一个假图
    x = torch.randn(1, 1, 128, 128, device=device, dtype=rcwa.rdtype)
    coeffs = net(x)  # [1,3]
    t, La, dc = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    ms2 = rcwa(t, La, dc)
    print(ms2.shape)
    loss2 = ms2.square().mean()
    loss2.backward()

    # 检查某层梯度
    g = net.backbone[0].weight.grad
    print("  conv1.weight.grad is None?", g is None)
    if g is not None:
        print("  conv1.weight.grad abs-mean:", float(g.abs().mean()))
