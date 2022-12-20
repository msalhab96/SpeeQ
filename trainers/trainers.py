from typing import Union
from data.interfaces import IDataLoader
from .interfaces import ISchedular, ITrainer
from constants import HistoryKeys
from torch.optim import Optimizer
from torch.nn import Module


class BaseTrainer(ITrainer):
    """Builds the basic trainer module

    Args:
        optimizer (Union[Optimizer, ISchedular]): The optimizer or the wrapped
        optimizer that will be used during the training.
        criterion (Module): The loss fucntion that will be used
        during the training process.
        model (Module): The model.
        train_loader (ILoader): The training data loader.
        test_loader (ILoader): The testing data loader.
        epochs (int): The number of epochs.
        log_steps_frequency (int): The number of steps to log the
        results after.
        history (dict): The history of the training if there is
        any. Default {}.
    """
    def __init__(
            self,
            optimizer: Union[Optimizer, ISchedular],
            criterion: Module,
            model: Module,
            train_loader: IDataLoader,
            test_loader: IDataLoader,
            epochs: int,
            log_steps_frequency: int,
            history: dict = {}
            ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.log_steps_frequency = log_steps_frequency
        self.history = history
        if HistoryKeys.test_loss not in self.history:
            self.history[HistoryKeys.test_loss] = list()
        if HistoryKeys.train_loss not in self.history:
            self.history[HistoryKeys.train_loss] = list()
