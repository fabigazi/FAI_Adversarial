from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from classify_model import LeNet5
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from target_model import TargetModel

def PGD(model, image, label, epsilon, iterations, alpha):

	perturbed_image = image.clone().detach().requires_grad_(True)

	for i in range(iterations):
		criterion = nn.CrossEntropyLoss()

		output = model(perturbed_image)

		loss = criterion(output, label)
		loss.backward()

		with torch.no_grad():
			perturbed_image.data = perturbed_image.data + alpha * perturbed_image.grad.sign()
			perturbed_image.data = torch.clamp(perturbed_image.data, image.data - epsilon, image.data + epsilon)

		perturbed_image.grad.zero_()

	return perturbed_image

model = LeNet5()
model.load_state_dict(torch.load("model_1.pth"))
model.eval()

train = datasets.MNIST(
    root="dataset/",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)
train_loader = DataLoader(train, batch_size=64, shuffle=True)

# print(max(train[0][0].flatten()))
target_model = TargetModel(model)

epsilon = 0.05
iterations = 500
alpha = 2/255

results = []

for batch, (X, y) in enumerate(train_loader):
	attack = PGD(model, X, y, epsilon, iterations, alpha)
	results.append(attack)
	break

print(results[0][0].shape)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(train[0][0].squeeze(0), cmap="gray")
ax.set_title("Original Image " + "pred: " + str(target_model.predict(train[0][0])))
ax = fig.add_subplot(1, 2, 2)
ax.imshow((train[0][0] + attack[0][0].detach().numpy()).squeeze(0), cmap="gray")
ax.set_title(
    "Poisoned Image "
    + "pred: "
    + str(target_model.predict(train[0][0] + attack[0][0].detach().numpy()))
)
plt.show()

