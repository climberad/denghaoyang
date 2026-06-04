import torch
import torch
# 输入（4个样本）
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])
# 真实值
y_true = torch.tensor([[3.0],
                       [5.0],
                       [7.0],
                       [9.0]])

W = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)
lr = 0.01

#前向传播，算损失，反向传播更新参数，继续计算
for epoch in range(1000):
    # forward
    y_pred = x @ W + b

    # loss（均方误差）
    loss = ((y_pred - y_true) ** 2).mean()

    # backward
    loss.backward()
    # 打印梯度：当x是当前值的时候，W和b的变化情况，正负表示更新方向，大小表示更新的步长
    if epoch % 100 == 0:
        print("W.grad:", W.grad)
        print("b.grad:", b.grad)
    # 更新参数（手动梯度下降）

    with torch.no_grad():  #告诉python无需记录这个计算，告诉python只改参数，无需记录
        W -= lr * W.grad
        b -= lr * b.grad

    # 清空梯度,要不然下次会自动累加。需要的是当前x的时候，w和b的偏导情况而不是加起来
    W.grad.zero_()
    b.grad.zero_()

print("更新后 W:", W)
print("更新后 b:", b)
