import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from utils import Normalize

import warnings

warnings.filterwarnings("ignore")


class TargetModel:
    def __init__(self, model):
        self.model = model

    def predict(self, sample):
        # preprocess = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #     ]
        # )
        # sample_image_tensor = preprocess(sample)

        # normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.model.eval()
        # pred = self.model(normalizer(sample_image_tensor))
        pred = self.model(sample.unsqueeze(0))
        return pred.max(dim=1)[1].item()
