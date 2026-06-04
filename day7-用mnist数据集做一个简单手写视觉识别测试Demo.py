import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from PIL import ImageOps
from torchvision import transforms

import tkinter as tk
from PIL import Image, ImageDraw
# 图像预处理，必须先转成Tensor。因为PyTorch的模型只能处理Tensor，但是原来的图片一般是图片对象
#还有调整维度顺序为C H W
transform = transforms.ToTensor()   #本来像素范围是0-255，这样一转，按比例变成0-1的范围




# 训练集
train_dataset = datasets.MNIST(
    root="./data",          #会在当前目录创建./data里面会保存MNIST数据文件
    train=True,             #训练集60000张
    download=True,          #如果本地没有就自动下载
    transform=transform    #每次读取图片的时候，自动换成Tensor
)

# 测试集
test_dataset = datasets.MNIST(
    root="./data",
    train=False,            #测试集10000张
    download=True,
    transform=transform
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,          #一次性训练64张图片
    shuffle=True            #打乱顺序，防止模型背答案
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

#images,labels,相当于输入x输出y
#tarin_loader是一个可迭代的对象，因为它从train_dataset里面一次性取出64张图。
#iter会把一个可迭代对象转换成为迭代器，迭代器会记住当前位置，然后每次next会返回下一批数据
images, labels = next(iter(train_loader))  #取训练集的第一批图片和标签
print(images.shape)
print(labels.shape)
#image是一个Tensor [64, 1, 28, 28]  [64, C, H, W]
#lable是一个数据标签

#定义CNN网络
#输入图像 -> 卷积 -> ReLU -> 池化 -> 卷积 -> ReLU -> 池化 -> Flatten -> FC -> 输出
# import torch
# import torch.nn as nn
#
# class CNN_Debug(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2)
#         self.fc = nn.Linear(8*14*14, 10)
#
#     def forward(self, x):
#         print("Input:", x.shape)  # 输入图片 shape
#
#         x = self.conv1(x)
#         print("After conv1:", x.shape)  # 卷积后
#
#         x = self.relu(x)
#         print("After ReLU:", x.shape)  # ReLU 不改变 shape
#
#         x = self.pool(x)
#         print("After MaxPool:", x.shape)  # 池化后
#
#         x = x.reshape(x.size(0), -1)
#         print("After flatten:", x.shape)  # 展平后
#
#         x = self.fc(x)
#         print("After fc:", x.shape)  # 输出层
#
#         return x
#可以看见
# Input: torch.Size([64, 1, 28, 28])
# After conv1: torch.Size([64, 8, 28, 28])
# After ReLU: torch.Size([64, 8, 28, 28])
# After MaxPool: torch.Size([64, 8, 14, 14])
# After flatten: torch.Size([64, 1568])
# After fc: torch.Size([64, 10])
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #对齐都是内部处理的不管，会用就行
        self.conv1 = nn.Conv2d(     #image是一个Tensor [64, 1, 28, 28]  [64, C, H, W]
            in_channels=1,          #输入一个通道 [64, 1, 28, 28] 对应这里第二项
            out_channels=8,         #8个卷积核，输出8个特征图，可以理解为，能够学习8种特征 输出第二项变成8 [64, 8, 28, 28]
            kernel_size=3,          #每个卷积核1*3*3，整个图片复用这8个卷积核，也就1*3*3*8个权重参数，每个卷积核一个bias所以是8个bias参数
            padding=1               #28*28 = 784， 3*3*87 = 783 所以要在最后补一个像素。
        )

        self.relu = nn.ReLU()       #加入非线性
                                        #取一个小区域的最大值
        self.pool = nn.MaxPool2d(2)     #把28*28 变成 14*14，用2*2的窗口，在窗口里面选最大的一个值

        self.fc = nn.Linear(8 * 14 * 14, 10)    #全连接层输出

    def forward(self, x):
        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.reshape(x.size(0), -1)        #展平方便全连接层

        x = self.fc(x)

        return x


# 假设你要处理图像任务，比如识别手写数字 0-9。
#
# 线性模型（Linear / Logistic Regression）做法：
# 把一张图片展平成一个长向量。
# 每个像素是一个特征，模型学一个权重对应每个像素。
# 输出是一个分类概率（比如哪个数字）。
#
# 问题：
#
# 参数太多：28x28 的灰度图像，展平后就是 784 个输入，如果是彩色图 32x32x3 = 3072 个输入。对大图像来说，参数量会爆炸。
# 空间结构信息丢失：线性模型把图像拉成一条向量，邻近的像素关系被破坏了。图像里的“局部模式”（比如边缘、角落、纹理）丢失了。
# 泛化能力差：如果一个数字移动了位置，线性模型很难正确识别，因为它对空间平移不敏感。
#
# 所以我们需要一种 能保留空间结构，同时参数可控的模型，这就是 CNN。
#初始化
model = CNN()

# loss_fn = nn.CrossEntropyLoss()     #损失函数整批[0,10]算loss
#
# optimizer = torch.optim.Adam(       #优化器CrossEntropyLoss，分类任务用这玩意
#     model.parameters(),
#     lr=0.001
# )

#完整循环
# for epoch in range(5):
#
#     model.train()
#
#     for images, labels in train_loader:
#
#         outputs = model(images)
#
#         loss = loss_fn(outputs, labels)
#
#         optimizer.zero_grad()
#
#         loss.backward()
#
#         optimizer.step()
#         # print(f"outputs: {outputs.shape}") [0,10]
#     # print(f"epoch {epoch}, loss={loss.item():.4f}")
#
# #测试准确率
# model.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#
#     for images, labels in test_loader:
#
#         outputs = model(images)
#
#         preds = outputs.argmax(dim=1)            #得到预测值,找列里面最大值的位置
#
#         correct += (preds == labels).sum().item()
#
#         total += labels.size(0)
#         # print(f"preds: {preds.shape}")
#         # print(f"outputs: {outputs.shape}")
# accuracy = correct / total
#
# print("accuracy:", accuracy)

# torch.save(model.state_dict(), "mnist_cnn.pth")


#下面是手写识别
#加载模型
model = CNN()
model.load_state_dict(
    torch.load("mnist_cnn.pth")
)
model.eval()
#创建画板
canvas_width = 280
canvas_height = 280
root = tk.Tk()
canvas = tk.Canvas(
    root,
    width=canvas_width,
    height=canvas_height,
    bg="white"
)
canvas.pack()
image = Image.new(
    "L",
    (canvas_width, canvas_height),
    "white"
)
draw = ImageDraw.Draw(image)

#实现画图
def paint(event):

    x = event.x
    y = event.y
    r = 10
    canvas.create_oval(
        x-r,
        y-r,
        x+r,
        y+r,
        fill="black",
        outline="black"
    )
    draw.ellipse(
        [x-r,y-r,x+r,y+r],
        fill="black"
    )
canvas.bind("<B1-Motion>", paint)

#图片预处理
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

result_label = tk.Label(
    root,
    text="预测结果：",
    font=("Arial", 12)
)
result_label.pack()

#预测函数
def predict():
    img = image.copy()
    img = ImageOps.invert(img)
    img = img.resize((28,28))
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1)
    result_label.config(
        text=f"预测结果：{pred.item()}"
    )

#clean函数
def clear_canvas():
    # 清空 Tkinter 画布
    canvas.delete("all")
    # 清空 PIL 图片
    draw.rectangle(
        [0, 0, canvas_width, canvas_height],
        fill="white"
    )



#按钮
frame = tk.Frame(root)
frame.pack()
predict_btn = tk.Button(
    frame,
    text="预测",
    command=predict
)
predict_btn.pack(side=tk.LEFT)
clear_btn = tk.Button(
    frame,
    text="清空",
    command=clear_canvas
)
clear_btn.pack(side=tk.LEFT)


#运行
root.mainloop()
