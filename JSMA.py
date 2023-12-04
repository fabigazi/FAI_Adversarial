from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from classify_model import LeNet5
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from target_model import TargetModel
import numpy as np
from art.attacks.evasion import SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier

model = LeNet5()
model.load_state_dict(torch.load("model_1.pth"))
model.eval()

classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=(1, 32, 32),
    nb_classes=10,
)

train = datasets.MNIST(
    root="dataset/",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)
train_loader = DataLoader(train, batch_size=64, shuffle=True)

target_model = TargetModel(model)

epsilon = 2./255

attack = SaliencyMapMethod(classifier, theta=0.1, gamma=1.0, batch_size=1, verbose=True)

for batch, (X, y) in enumerate(train_loader):
    original = X.numpy()
    out = attack.generate(x=original, y=y.numpy(), eps=epsilon)
    perturbed_image = torch.tensor(out)
    # with torch.no_grad():
    #     original_output = target_model.predict(torch.tensor(original[0][0]))
    #     perturbed_output = target_model.predict(original[0][0] + perturbed_image[0][0].detach().numpy())
    # print("Original Predicted Class:", original_output)
    # print("Perturbed Predicted Class:", perturbed_output)
    break

noise = torch.tensor(original[0][0]) - perturbed_image[0][0]
# if torch.tensor(original[0][0]) == perturbed_image[0][0]:
#     print("sfvusvj")

# print(noise)
fitness = 50 - (0.5 * np.linalg.norm(noise.detach().numpy()))
print(fitness)

newimage = train[0][0] + perturbed_image[0][0].detach().numpy()
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(train[0][0].squeeze(0), cmap="gray")
ax.set_title("Original Image " + "pred: " + str(target_model.predict(train[0][0])))
ax = fig.add_subplot(1, 2, 2)
ax.imshow((newimage).squeeze(0), cmap="gray")
ax.set_title(
    "Poisoned Image "+ "pred: "+ str(target_model.predict(newimage))
)
plt.show()