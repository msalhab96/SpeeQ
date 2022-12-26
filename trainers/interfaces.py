from abc import ABC, abstractmethod, abstractproperty


class ITrainer(ABC):

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def test():
        pass

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def inline_log():
        pass

    @abstractproperty
    def is_master():
        pass


class ISchedular(ABC):

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def step(self):
        pass
