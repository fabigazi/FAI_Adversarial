import numpy as np
import pandas as pd
import json
from typing import Tuple
from PIL import Image
from torchvision import transforms
from utils import Normalize
from torchvision.models import resnet50
import torch
import torch.nn as nn
from target_model import TargetModel
from classify_model import LeNet5
import warnings
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


class Magma:
    def __init__(
        self,
        population_size: int,
        sample_shape: Tuple,
        num_evolutions: int,
        beta: float = 0.5,
        epsilon: float = 30.0 / 255,
    ) -> None:
        self.population_size = population_size
        self.sample_shape = sample_shape
        self.num_evolutions = num_evolutions
        self.beta = beta
        self.epsilon = epsilon
        self.parent_selection_rate = 0.2
        self.inheritance_rate = 0.7
        self.mutation_rate = 0.01
        self.population = self.initialize_population()

    def generate_candidate(self):
        return torch.tensor(
            np.random.uniform(
                low=-self.epsilon, high=self.epsilon, size=self.sample_shape
            ),
            dtype=torch.float32,
        )

    def sanitize_population(self, population):
        sanitized_population = []
        for i in range(len(population)):
            sanitized_population.append(
                np.clip(population[i], -self.epsilon, self.epsilon)
            )
        return sanitized_population

    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.generate_candidate())
        return population

    def fitness_untargetted(self, poison_sample, victim_sample, target_model):
        # print(poison_sample.float().dtype)
        # print(victim_sample.dtype)
        # print(poison_sample)
        # print(victim_sample)
        pred_poison = target_model.predict(poison_sample + victim_sample)
        pred_base = target_model.predict(victim_sample)
        missclassification_reward = 0
        if pred_poison != pred_base:
            missclassification_reward = 100
        poison_magnitude = np.linalg.norm(poison_sample)
        # print(poison_magnitude)
        fitness = (self.beta * missclassification_reward) - (
            (1 - self.beta) * poison_magnitude
        )

        return fitness

    def select_top_k(self, fitnesses, k):
        return np.argsort(fitnesses)[-k:]

    def generate_new_population_from_fit_candidates(
        self, fit_candidates, num_new_candidates
    ):
        new_population = []
        for i in range(num_new_candidates):
            new_candidate = self.generate_candidate()
            sample_fit_candidate_idx = np.random.choice(len(fit_candidates))
            sample_fit_candidate = fit_candidates[sample_fit_candidate_idx]
            # Sample 3 dimensional indices
            for channel in range(len(new_candidate)):
                for row in range(len(new_candidate[channel])):
                    for col in range(len(new_candidate[channel][row])):
                        if np.random.uniform() < self.inheritance_rate:
                            new_candidate[channel][row][col] = sample_fit_candidate[
                                channel
                            ][row][col]
            new_population.append(new_candidate)

        for c in fit_candidates:
            new_population.append(c)

        return new_population

    def mutate_candidate(self, candidate):
        new_candidate = candidate.clone()
        for i in range(len(new_candidate)):
            for j in range(len(new_candidate[i])):
                if np.random.uniform() < self.mutation_rate:
                    new_candidate[i][j] = np.random.uniform(-self.epsilon, self.epsilon)
        return new_candidate

    def attack(self, victim_sample, target_model):
        for e in range(self.num_evolutions):
            print("Evolution: ", e)
            fitnesses = []
            for candidate in self.population:
                fitnesses.append(
                    self.fitness_untargetted(candidate, victim_sample, target_model)
                )
            fit_candidates_indices = self.select_top_k(
                fitnesses, int(self.population_size * self.parent_selection_rate)
            )
            print([fitnesses[x] for x in fit_candidates_indices][:10])
            fit_candidates = [self.population[fc] for fc in fit_candidates_indices]
            new_population = self.generate_new_population_from_fit_candidates(
                fit_candidates, self.population_size - len(fit_candidates)
            )
            for i in range(len(new_population)):
                new_population[i] = self.mutate_candidate(new_population[i])
            self.population = self.sanitize_population(new_population)

        fitnesses = []
        for candidate in self.population:
            fitnesses.append(
                self.fitness_untargetted(candidate, victim_sample, target_model)
            )
        fit_candidates_indices = self.select_top_k(
            fitnesses, 1
        )  # Select the best candidate
        return self.population[fit_candidates_indices[0]]


# sample_img = Image.open("owl.jpeg")

# preprocess = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#     ]
# )
# sample_image_tensor = preprocess(sample_img)

# normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# model = resnet50(pretrained=True)
# model.eval()
# print("input to nn")
# print(normalizer(sample_image_tensor))
# pred = model(normalizer(sample_image_tensor))

# with open("imagenet_class_index.json") as f:
#     imagenet_classes = {int(i): x[1] for i, x in json.load(f).items()}
# with open("imagenet_class_index.json") as f:
#     imagenet_labels = {x[1]: int(i) for i, x in json.load(f).items()}

# print(imagenet_classes[pred.max(dim=1)[1].item()])
# print(pred.max(dim=1)[1].item())
# pred_class = pred.max(dim=1)[1].item()
# print(nn.CrossEntropyLoss()(pred, torch.LongTensor([pred_class])).item())

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
print(target_model.predict(train[0][0]))
# print(model(train[0][0].unsqueeze(0)).max(dim=1)[1].item())

attack = Magma(100, train[0][0].shape, 50)
attack_result = attack.attack(train[0][0], target_model)

# attack = Magma(20, transforms.ToTensor()(sample_img).shape, 1000)
# sample_candidate = attack.generate_candidate()

# target_model = TargetModel(resnet50(pretrained=True))
# attack_result = attack.attack(transforms.ToTensor()(sample_img), target_model)


# orig_img = Image.fromarray(
#     ((train[0][0] + attack_result).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
# )
# orig_img.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(train[0][0].squeeze(0), cmap="gray")
ax.set_title("Original Image " + "pred: " + str(target_model.predict(train[0][0])))
ax = fig.add_subplot(1, 2, 2)
ax.imshow((train[0][0] + attack_result).squeeze(0), cmap="gray")
ax.set_title(
    "Poisoned Image "
    + "pred: "
    + str(target_model.predict(train[0][0] + attack_result))
)
plt.show()
