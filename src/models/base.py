from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def loss(self, batch):
        pass

    @abstractmethod
    def predict(self, batch):
        pass
