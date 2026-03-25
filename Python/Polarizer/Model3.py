import torch
import torch.nn as nn



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    """
    兼容旧代码保留的占位类。
    当前文件已统一改为 Model_grating 的 CNN + GAP + MLP 架构，
    该残差块不再参与实际网络构建。
    """
    def __init__(self, in_channels=None, out_channels=None, stride=1, downsample=None):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)


class IoutCNNMLP(nn.Module):
    """
    输入:  Iout [N,256,256] 或 [N,1,256,256]
    输出:  [N,out_dim]
    结构:  Conv(降采样提特征) -> GAP -> MLP(回归)
    """
    def __init__(self, out_dim: int = 1, device: str = "cpu", dtype: torch.dtype = None):
        super().__init__()
        self.device = torch.device(device)

        # ====== CNN: 256 -> 128 -> 64 -> 32 -> 16 -> 8 ======
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.act3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.act4 = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.act5 = nn.LeakyReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ====== MLP: 256 -> 128 -> 64 -> out_dim ======
        self.fc1 = nn.Linear(256, 128, bias=True)
        self.act_fc1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(128, 64, bias=True)
        self.act_fc2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(64, out_dim, bias=True)

        if dtype is not None:
            self.to(device=self.device, dtype=dtype)
        else:
            self.to(device=self.device)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input with 3 or 4 dims, got shape {tuple(x.shape)}")

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))

        x = self.gap(x).flatten(1)

        x = self.act_fc1(self.fc1(x))
        x = self.act_fc2(self.fc2(x))
        out = self.fc3(x)
        return out


class ICmosToThetaResNet(nn.Module):
    """
    兼容旧接口保留的包装类。
    现在内部实际使用的是 Model_grating 同源的 IoutCNNMLP 架构。

    旧参数 block / layers / num_classes / out_eps / scale_max 会被忽略，
    这样下游脚本即使沿用旧调用方式也不会报错。
    """
    def __init__(
        self,
        block=None,
        layers=None,
        num_classes=121,
        out_eps: float = 1e-6,
        scale_max: float = 1000.0,
        out_dim: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = None,
        **kwargs,
    ):
        super().__init__()
        self.model = IoutCNNMLP(out_dim=out_dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.model(x)


def icmos_to_ms_resnet(layers=None, **kwargs):
    return ICmosToThetaResNet(ResidualBlock, layers, **kwargs)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [2, 2, 2, 2, 2]
    model = icmos_to_ms_resnet(layers, out_dim=1, device=device, dtype=torch.float32)
    model.apply(weights_init)

    x = torch.randn(10, 256, 256, device=device, dtype=torch.float32)
    y = model(x)
    print("输出尺寸:", y.shape)
