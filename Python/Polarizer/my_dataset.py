import torch
from torch.utils.data import Dataset
import os
import numpy as np


class MyDataset(Dataset):

    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_item = self.input_data[idx]
        label_item = self.labels[idx]

        return input_item, label_item


if __name__ == "__main__":

########################################################################################################################

    # 设置数据的根目录
    root_path = "../../Datasets/Database2/"
    # 加载训练数据
    inputs = torch.load(os.path.join(root_path, "I_out_train.pt"))
    labels = torch.load(os.path.join(root_path, "Ms_train.pt"))
    print(inputs.shape)
    print(labels.shape)


    train_data_set = MyDataset(inputs, labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                               batch_size=50,
                                               shuffle=True)
    train_loader_iter = iter(train_loader)
    x, y = next(train_loader_iter)
    print(x.shape)   # torch.Size([12, 3321])
    print(y.shape)   # torch.Size([12, 6])






