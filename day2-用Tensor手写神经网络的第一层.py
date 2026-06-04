import torch
#一个batch
# # 输入
# x = torch.tensor([1.0, 2.0])
# # 权重
# W = torch.tensor([[1.0, 2.0],
#                   [3.0, 4.0]])
# # 偏置
# b = torch.tensor([1.0, 1.0])
# # 计算
# y = torch.matmul(W, x) + b
# print(y)

#两个batch
x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])  # 2个样本
W = torch.tensor([[2.0, 4.0],   #在Tensor里面W的行数是神经元 Pytorch里面默认W(输出，输入)
                  [6.0, 8.0],
                  [9.0, 10.0]])
b = torch.tensor([1.0, 1.0, 1.0])

# 如果你把 W 定义成 (输出, 输入)，那“行是神经元”
# 如果你把 W 定义成 (输入, 输出)，那“列是神经元”
y = torch.matmul(x, W.T) + b   #这里属于w是每一行是一个神经元，所以现需要转置
print(y)
print(b.shape)


#每一个样本通过加权求和再加偏置，是按行排列的。权重每一列就是一个神经元