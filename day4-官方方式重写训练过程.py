import torch
import torch.nn as nn
import torch.optim as optim

# 数据
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])

y_true = torch.tensor([[3.0],
                       [5.0],
                       [7.0],
                       [9.0]])


#面向对象思想，创建model和loss_fn以及optimizer对象，用来实现forward，loss，backward，update操作
# 模型
model = nn.Linear(1, 1) #就是y_pred = x @ W + b

# 损失函数
loss_fn = nn.MSELoss()      #就是((y_pred - y_true) ** 2).mean()

# 优化器 记住所有参数，存学习率，负责更新参数w，b
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练
for epoch in range(20):
    # forward
    y_pred = model(x)       #model相当于nn.linear创建的一个实例，实例实现了__call__方法

    # loss
    loss = loss_fn(y_pred, y_true)  #同样也是创建了实例，实现了call方法

    # backward
    optimizer.zero_grad()       #清空梯度，要不然会累加
    loss.backward()             #自动求导参数在y_pred里面，pred和loss有关，所以这个就是自动求梯度

    # update，更新参数w，b
    optimizer.step()

    print(f"epoch {epoch}, loss = {loss.item():.4f}")
    print(model.weight.item())
    print(model.bias.item())