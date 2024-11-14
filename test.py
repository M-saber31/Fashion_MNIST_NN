import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor(),)
classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal", "Shirt","Sneaker","Bag", "Ankle Boot"]

model.eval()

X, y = test_data[0][0], test_data[0][1]

with torch.no_grad():
    pred = model(X)
    predicted = classes[pred[0].argmax(0)]
    actual = classes[y]
    print("Predicted: ", predicted, " Actual is ", actual)
