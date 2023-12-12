from torchvision import transforms
from torchvision import datasets
import torch
from classify_model import LeNet5
from target_model import TargetModel
from torch.utils.data import DataLoader

#load model
model = LeNet5()
model.load_state_dict(torch.load("model_1.pth"))
model.eval()

target_model = TargetModel(model)

#test data
test = datasets.MNIST(
    root="dataset/",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)
test_loader = DataLoader(test, batch_size=64, shuffle=False)

#evaluate model
model.eval() 
correct = 0
with torch.no_grad(): 
    for X, y in test_loader: 
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

correct /= len(test_loader.dataset)
print(f"Accuracy on test: {(100*correct):>0.1f}%")