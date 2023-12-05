from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from classify_model import LeNet5
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from target_model import TargetModel
import numpy as np
import imageio

gif_images = torch.tensor([])

gif_images = []


def PGD(model, image, label, epsilon, iterations, alpha):
    perturbed_image = image.clone().detach().requires_grad_(True)
    gif_images.append(perturbed_image[0].detach().numpy())

    for i in range(iterations):
        # gif_images = torch.cat((gif_images, perturbed_image[0]), 0)
        criterion = nn.CrossEntropyLoss()
        output = model(perturbed_image)
	for i in range(iterations):
		# gif_images = torch.cat((gif_images, perturbed_image[0]), 0)
		criterion = nn.CrossEntropyLoss()
		output = model(perturbed_image)

	for i in range(iterations):
		# gif_images = torch.cat((gif_images, perturbed_image[0]), 0)
		criterion = nn.CrossEntropyLoss()
		output = model(perturbed_image)

        with torch.no_grad():
            perturbed_image.data = (
                perturbed_image.data + alpha * perturbed_image.grad.sign()
            )
            perturbed_image.data = torch.clamp(
                perturbed_image.data, image.data - epsilon, image.data + epsilon
            )
            # perturbed_image = torch.clamp(perturbed_image, 0, 1)

        perturbed_image.grad.zero_()
        perturbed_image[0][0]
        gif_images.append(perturbed_image[0].detach().numpy())

    return perturbed_image


# model = LeNet5()
# model.load_state_dict(torch.load("model_1.pth"))
# model.eval()

# train = datasets.MNIST(
#     root="dataset/",
#     train=False,
#     download=True,
#     transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
# )
# train_loader = DataLoader(train, batch_size=64, shuffle=False)

# # print(max(train[0][0].flatten()))
# target_model = TargetModel(model)

# epsilon = 0.05
# iterations = 50
# alpha = 2.0 / 255
# print(len(train_loader) * 64)
# i = 0

# new_dataset = torch.tensor([])
# for batch, (X, y) in enumerate(train_loader):
#     original = X
#     attack = PGD(model, X, y, epsilon, iterations, alpha)
#     new_dataset = torch.cat((new_dataset, attack), 0)
#     # i += 1
#     # # print(i)
#     # if i > 10:
#     # 	break
#     break

# # Andrew use this!
# # gif_images = gif_images.detach().numpy()

# # convert images to GIF
# from PIL import Image

# imgs = [
#     Image.fromarray(np.squeeze(img.transpose(1, 2, 0), axis=2) * 255)
#     for img in gif_images
# ]
# imgs[0].save(
#     "pgd_progress.gif", save_all=True, append_images=imgs[1:], duration=150, loop=100
# )


# # change this to try different examples by their index.
# figure_ind = 6

# noise = original[figure_ind][0] - new_dataset[figure_ind][0]
# fitness = 50 - (0.5 * np.linalg.norm(noise.detach().numpy()))

# print(fitness)


# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(train[figure_ind][0].squeeze(0), cmap="gray")
# ax.set_title(
#     "Original Image " + "pred: " + str(target_model.predict(train[figure_ind][0]))
# )
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow((new_dataset[figure_ind][0].detach().numpy()), cmap="gray")
# ax.set_title(
#     "Poisoned Image " + "pred: " + str(target_model.predict(new_dataset[figure_ind]))
# )
# plt.show()
