import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import tempfile
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataset
from Model3 import ResidualBlock, ICmosToThetaResNet, icmos_to_ms_resnet, weights_init
from theta2Ms import M_pol


def count_conv_layers(model):
    conv_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers += 1
            # print(f"{name}: {module}")
    print(f"\n总的卷积层数: {conv_layers}")


"""加载模型"""
layers = [2, 2, 2, 2, 2]
# 创建模型实例
model = icmos_to_ms_resnet(layers)

# 初始化参数
model.apply(weights_init)

"""加载数据"""
root_path = "../../Datasets/Database2/"
# 加载训练数据
I_cmos = torch.load(os.path.join(root_path, "I_out_train.pt"))
print(I_cmos.shape)

Ms = torch.load(os.path.join(root_path, "Ms_train.pt"))
print(Ms.shape)

train_data_set = MyDataset(I_cmos, Ms)
train_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                           batch_size=120,
                                           shuffle=True)


"""加载测试数据"""
I_cmos_test = torch.load(os.path.join(root_path, "I_out_test.pt"))
print(I_cmos_test.shape)

Ms_test = torch.load(os.path.join(root_path, "Ms_test.pt"))
print(Ms_test.shape)

test_data_set = MyDataset(I_cmos_test, Ms_test)
test_loader = torch.utils.data.DataLoader(dataset=test_data_set,
                                          batch_size=I_cmos_test.shape[0],
                                          shuffle=False)

"""开始训练"""
# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 丢到GPU上去
model = model.double().to(device)
TORCH_SAVE = torch.save(model.state_dict(), './Saved_model/model_parameters.pth')
# 定义损失函数和优化器
# 使用均方误差损失
criterion = nn.MSELoss()
learning_rate = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备保存最佳模型的变量
best_loss = float('inf')  # 初始化最佳损失为正无穷大
best_epoch = -1  # 初始化最佳 epoch

# 训练循环
num_epochs = 30000
training_loss = []
testing_loss = []  # 新增测试集损失列表

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs_batch, targets_batch) in enumerate(train_loader):
        inputs_batch = inputs_batch.to(device)     # [batch, 256, 256]
        targets_batch = targets_batch.to(device)   # [batch, 241, 4, 4]
        optimizer.zero_grad()
        theta = model(inputs_batch)                # [batch, 241, 1]
        # **应用激活函数限制 theta 的范围在 [0, pi]**
        outputs = M_pol(theta)                     # [batch, 241, 4, 4]

        loss = criterion(outputs, targets_batch)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.6f}")

    epoch_loss = running_loss / len(train_loader)
    training_loss.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.6f}")

    # 检查是否为当前最佳模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch + 1  # epoch 从 0 开始，需要加 1
        torch.save(model.state_dict(), './Saved_model/best_model_parameters.pth')
        print(f"Best model saved at epoch {best_epoch} with loss {best_loss:.6f}")

    # 每 10 个 epoch 在测试集上验证一次
    if (epoch + 1) % 10 == 0:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs_batch, targets_batch in test_loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)

                theta = model(inputs_batch)  # [batch, 241, 1]
                # **应用激活函数限制 theta 的范围在 [0, pi]**
                theta = torch.sigmoid(theta) * math.pi  # 将 theta 映射到 [0, π] 范围 # [batch, 241, 1]
                outputs = M_pol(theta)  # [batch, 241, 4, 4]
                loss = criterion(outputs, targets_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        testing_loss.append(test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss:.6f}")



# 训练完成后，保存模型参数
print("model saved!")

# 保存 training_loss 列表
torch.save(training_loss, './Saved_model/training_loss.pt')
torch.save(testing_loss, './Saved_model/testing_loss.pt')
print("Training loss saved!")