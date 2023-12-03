import torch
import torch.nn as nn

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