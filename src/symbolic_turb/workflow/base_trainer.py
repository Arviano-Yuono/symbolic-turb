from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train(self):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the model."""
        pass
