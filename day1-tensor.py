import torch
import numpy as np
x = torch.tensor(5)
print(x.shape)  # torch.Size([])

x = torch.tensor([5,6,7,8])     #属于没有行列概念的一维张量，就是四列就对了
print(x.shape)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.shape)  # torch.Size([2, 3])

x = torch.randn(2, 3, 4)
print(x.shape)

#Tensor和Numpy可以相互转换
# a = np.array([1, 2, 3])
# b = torch.tensor(a)
#
# c = b.numpy()

#Tensor可以自动求导
# x = torch.tensor(2.0, requires_grad=True)
# y = x**2
# # y.backward()
# # print(x.grad)     #一阶导（梯度）
# # y.backward()      #直接报错，默认求导之后会把求导图删掉
# # print(x.grad)
#
# #二阶导
# grad1 = torch.autograd.grad(y, x,create_graph=True)[0]
# grad2 = torch.autograd.grad(grad1, x)[0]
# print(grad2)

