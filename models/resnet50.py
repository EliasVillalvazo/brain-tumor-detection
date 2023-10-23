import torchvision.models as models
import torch.nn as nn

from models import base

class ResNet50(base.BaseModel):
    def __init__(self, learning_rate: float, pretrained: bool, num_classes: int=2, seed:int=None):
        """

        :param learning_rate:
        :type learning_rate:
        :param pretrained:
        :type pretrained:
        :param num_classes:
        :type num_classes:
        :param seed:
        :type seed:
        """
        super().__init__(learning_rate, seed=seed)

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        self.resnet.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                            nn.Linear(9216, 4096),
                                            nn.ReLU(), nn.Dropout(p=0.5),
                                            nn.Linear(4096, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.resnet(x)