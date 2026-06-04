import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(-2, 2, 100).reshape(-1, 1)
        self.y = self.x**2

    def __len__(self):      #返回数据集有多少个样本
        return len(self.x)

    def __getitem__(self, idx):  #按索引招样本
        return self.x[idx], self.y[idx]

dataset = MyDataset()

# batch 切分
# shuffle 打乱
# 多线程读取
# 自动迭代
loader = DataLoader(
    dataset,                #数据集
    batch_size=16,          #每次取多少个样本，总共100个，每次取16个，最后剩下4个单独取
    shuffle=True            #每个 epoch 开始前，随机打乱数据
)


model = TwoLayerNet()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    model.train()  # 训练阶段打开这个，学习阶段

    for batch_x, batch_y in loader:     #100个样本分批次训练，每批16个x对应标准值y，每循环一次更新一次权重和偏置
        y_pred = model(batch_x)

        loss = loss_fn(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print(f"epoch {epoch}, loss={loss.item():.4f}")

# torch.save(model.state_dict(), "model.pth")     #保存训练参数

#有时候还要保存
# optimizer状态
# epoch
# loss
#标准写法如下
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "checkpoint.pth")

#加载
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(
    checkpoint['model_state_dict']
)

optimizer.load_state_dict(
    checkpoint['optimizer_state_dict']
)
#保存optimizer的原因是因为：
#要保存当前梯度和历史动量，要不然训练不会连续
#历史动量：小球下山的整体运动方向，Momentum和传统SGD不一样，会根据历史的整体方向
#决定参数的更新方向，从而减少参数更新的震动


model.eval()     #测试的时候打开这个，考试阶段
with torch.no_grad():   #已经训练完了，现在测试，测试的时候不用backward，所以关掉梯度，能节省显存
    y_pred = model(dataset.x)
    loss = loss_fn(y_pred, dataset.y)
print("eval loss:", loss.item())


# model_1 = TwoLayerNet() #重新创建model_1
#
# model_1.load_state_dict(        #加载出刚刚训练好的参数
#     torch.load("model.pth")
# )
#
# model.eval()                    #开启测试模式，实验结果和之前刚训练完测试的时候是一样的
# with torch.no_grad():
#     y_pred_1 = model_1(dataset.x)
#     loss_1 = loss_fn(y_pred_1, dataset.y)
# print("eval loss 1:", loss_1.item())