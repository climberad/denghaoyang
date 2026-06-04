model = CNN()

model.load_state_dict(
    torch.load("mnist_cnn.pth")
)

model.eval()