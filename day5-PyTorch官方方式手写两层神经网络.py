import torch
import torch.nn as nn
import torch.optim as optim

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

x = torch.linspace(-2, 2, 100).reshape(-1, 1)
y = x ** 2

model = TwoLayerNet()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("=== before training ===")
for name, param in model.named_parameters():
    print(name, param.data)

for epoch in range(200):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss={loss.item():.4f}")
        # print(model.weight.item())
        # print(model.bias.item())

print("=== after training ===")
for name, param in model.named_parameters():
    print(name, param.data)