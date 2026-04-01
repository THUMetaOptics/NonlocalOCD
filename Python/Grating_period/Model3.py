import torch
import torch.nn as nn


class IoutCNNMLP(nn.Module):
    """
    输入:  Iout [N,256,256] 或 [N,1,256,256]
    输出:  [N,3] （你可以解释为 [d, A, B] 或 [d, Lambda, duty] 等）
    结构:  Conv(降采样提特征) → GAP → MLP(回归)
    """
    def __init__(self, out_dim: int = 1, device: str = "cpu", dtype: torch.dtype = None):
        super().__init__()
        self.device = torch.device(device)

        # ====== CNN: 256 -> 128 -> 64 -> 32 -> 16 -> 8 ======
        self.conv1 = nn.Conv2d(1,   32, kernel_size=7, stride=2, padding=3, bias=False)  # 256->128
        self.act1  = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1, bias=False)  # 128->64
        self.act2  = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)  # 64->32
        self.act3  = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1, bias=False)  # 32->16
        self.act4  = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, bias=False)  # 16->8
        self.act5  = nn.LeakyReLU(inplace=True)

        self.gap   = nn.AdaptiveAvgPool2d(1)  # [N,256,8,8] -> [N,256,1,1]（只保留通道统计）

        # ====== MLP: 256 -> 128 -> 64 -> out_dim ======
        self.fc1   = nn.Linear(256, 128, bias=True)
        self.act_fc1 = nn.LeakyReLU(inplace=True)

        self.fc2   = nn.Linear(128, 64, bias=True)
        self.act_fc2 = nn.LeakyReLU(inplace=True)

        self.fc3   = nn.Linear(64, out_dim, bias=True)

        # 把整个模型搬到目标 device / dtype
        if dtype is not None:
            self.to(device=self.device, dtype=dtype)
        else:
            self.to(device=self.device)

    def forward(self, x):
        # 接受 [N,256,256] 或 [N,1,256,256]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # ---- CNN ----
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        # [N,256,8,8] -> [N,256]
        x = self.gap(x).flatten(1)
        # ---- MLP ----
        x = self.act_fc1(self.fc1(x))
        x = self.act_fc2(self.fc2(x))
        out = self.fc3(x)  # [N,3]，原始实数回归输出
        return out


if __name__ == "__main__":

    device = "cuda"  # 或 "cpu"
    model = IoutCNNMLP(out_dim=3, device=device, dtype=torch.float32)
    # 可选：你的 kaiming 初始化
    # model.apply(weights_init)

    Iout = torch.randn(4, 256, 256, device=device, dtype=torch.float32)
    y = model(Iout)  # [4,3]
    print(y.shape)