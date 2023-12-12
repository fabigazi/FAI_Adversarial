import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import tqdm
from classify_model import LeNet5
from PGD_attack import PGD

train = datasets.MNIST(root='dataset/', train=True, download=True, 
                       transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()]))
test = datasets.MNIST(root='dataset/', train=False, download=True,
                      transform=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()]))

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.001
num_epochs = 10
epsilon = 0.075

#Load old network
pgd_model = LeNet5().to(device)
pgd_model.load_state_dict(torch.load("model_1.pth"))
pgd_model.eval()

cost = nn.CrossEntropyLoss()
optim = torch.optim.Adam(pgd_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs): 
    for batch, (X, y) in enumerate(train_loader): 
        X = X.to(device)
        y = y.to(device)

        X = PGD(pgd_model, image=X, label=y)

        pred = pgd_model(X)
        loss = cost(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step() 

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")


pgd_model.eval() 
correct = 0
with torch.no_grad(): 
    for X, y in test_loader: 
        pred = pgd_model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

correct /= len(test_loader.dataset)
print(f"Accuracy on test: {(100*correct):>0.1f}%")

torch.save(pgd_model.state_dict(), "model_2_PGD.pth")