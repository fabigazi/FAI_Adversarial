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
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Magma:
    def __init__(
        self,
        population_size: int,
        sample_shape: Tuple,
        num_evolutions: int,
        beta: float = 0.5,
        epsilon: float = 40.0 / 255,
        targeted: bool = False,
    ) -> None:
        self.population_size = population_size
        self.sample_shape = sample_shape
        self.num_evolutions = num_evolutions
        self.beta = beta
        self.epsilon = epsilon
        print(self.num_evolutions)
        print(self.population_size)
        print(self.sample_shape)
        self.parent_selection_rate = 0.2
        self.inheritance_rate = 0.85
        self.mutation_rate = 0.3
        self.max_norm = np.linalg.norm(np.full(fill_value=1, shape=self.sample_shape))
        self.targeted = targeted
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
        # pred_poison = target_model.predict(poison_sample + victim_sample)
        # pred_base = target_model.predict(victim_sample)
        pred_poison_logits = target_model.predict_logits(poison_sample + victim_sample)
        pred_base_logits = target_model.predict_logits(victim_sample)
        abs_diff = np.abs(pred_poison_logits - pred_base_logits)
        missclassification_score = np.sum(abs_diff) / len(abs_diff)
        poison_magnitude = np.linalg.norm(poison_sample) / self.max_norm
        fitness = (self.beta * missclassification_score) - (
            (1 - self.beta) * poison_magnitude
        )

        return fitness

    def fitness_targetted(
        self, poison_sample, victim_sample, target_model, target_class
    ):
        pred_base = target_model.predict(victim_sample)
        pred_poison_logits = target_model.predict_logits(poison_sample + victim_sample)
        # pred_base_logits = target_model.predict_logits(victim_sample)
        # missclassification_score = (
        #     np.abs(1 - pred_poison_logits[pred_base])
        #     + np.abs(pred_poison_logits[target_class])
        # ) / 2
        missclassification_score = np.abs(pred_poison_logits[target_class])

        poison_magnitude = np.linalg.norm(poison_sample) / self.max_norm
        fitness = (self.beta * missclassification_score) - (
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
            random_candidate = self.generate_candidate()
            sample_fit_candidate = fit_candidates[np.random.choice(len(fit_candidates))]
            new_candidate = (self.inheritance_rate * sample_fit_candidate) + (
                1 - self.inheritance_rate
            ) * random_candidate
            new_population.append(self.mutate_candidate(new_candidate))

        return new_population

    def mutate_candidate(self, candidate):
        new_candidate = candidate.clone()
        for i in range(len(new_candidate)):
            for j in range(len(new_candidate[i])):
                if np.random.uniform() < self.mutation_rate:
                    new_candidate[i][j] = np.random.uniform(-self.epsilon, self.epsilon)
        return new_candidate

    def attack(self, victim_sample, target_model, target_class=None):
        # pbar = tqdm(total=self.num_evolutions)
        # for e in pbar:
        for e in range(1, self.num_evolutions + 1):
            if e % 100 == 0:
                print("Evolution: ", e)
            if e % 250 == 0:
                self.epsilon = self.epsilon * 0.95
            fitnesses = []
            if self.targeted:
                for candidate in self.population:
                    fitnesses.append(
                        self.fitness_targetted(
                            candidate, victim_sample, target_model, target_class
                        )
                    )
            else:
                for candidate in self.population:
                    fitnesses.append(
                        self.fitness_untargetted(candidate, victim_sample, target_model)
                    )
            fit_candidates_indices = self.select_top_k(
                fitnesses, int(self.population_size * self.parent_selection_rate)
            )
            # print([fitnesses[x] for x in fit_candidates_indices][-10:])
            if e % 100 == 0:
                print(
                    "Evolution: ", e, "Fittest: ", fitnesses[fit_candidates_indices[-1]]
                )
            fit_candidates = [self.population[fc] for fc in fit_candidates_indices]
            new_population = self.generate_new_population_from_fit_candidates(
                fit_candidates, self.population_size - len(fit_candidates)
            )
            if e % 10 == 0:
                fig = plt.figure()
                plt.imshow((victim_sample + fit_candidates[-1]).squeeze(0), cmap="gray")
                plt.savefig("./genetic/untargetted/{}.png".format(e))
                plt.close()
            self.population = fit_candidates + new_population

        fitnesses = []
        for candidate in self.population:
            fitnesses.append(
                self.fitness_untargetted(candidate, victim_sample, target_model)
            )
        fit_candidates_indices = self.select_top_k(fitnesses, 1)
        return self.population[fit_candidates_indices[0]]


model = LeNet5()
model.load_state_dict(torch.load("model_1.pth"))
model.eval()

train = datasets.MNIST(
    root="dataset/",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
)
# train_loader = DataLoader(train, batch_size=64, shuffle=True)

target_model = TargetModel(model)
# print(target_model.predict_logits(train[0][0]))
# print(model(train[0][0].unsqueeze(0)))

select_idx = 0
attack = Magma(500, train[select_idx][0].shape, 5000)
attack_result = attack.attack(train[select_idx][0], target_model, 6)


# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(train[select_idx][0].squeeze(0), cmap="gray")
# ax.set_title("Original Image " + "pred: " + str(target_model.predict(train[0][0])))
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow((train[select_idx][0] + attack_result).squeeze(0), cmap="gray")
# ax.set_title(
#     "Poisoned Image "
#     + "pred: "
#     + str(target_model.predict(train[select_idx][0] + attack_result))
# )
# plt.show()

# print(target_model.predict(train[select_idx][0] + attack_result))
# print(target_model.predict_logits(train[select_idx][0] + attack_result))
# print(attack.fitness_untargetted(attack_result, train[select_idx][0], target_model))
