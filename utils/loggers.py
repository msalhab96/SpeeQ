from pathlib import Path
from typing import Union
from .utils import clear
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


class ILogger(ABC):

    @abstractmethod
    def log_step(self):
        pass

    @abstractmethod
    def log(self):
        pass


class TBLogger(ILogger):
    def __init__(
            self,
            log_dir: Union[str, Path],
            n_logs: int,
            clear_screen: bool,
            *args,
            **kwargs
            ) -> None:
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self.__counters = dict()
        self.n_logs = n_logs
        self.clear_screen = clear_screen
        print('Started!')

    def log_step(
            self,
            key: str,
            category: str,
            value: Union[int, float]
            ) -> None:
        tag = f'{key}/{category}'
        if tag in self.__counters:
            self.__counters[tag] += 1
            counter = self.__counters[tag]
        else:
            self.__counters[tag] = 0
            counter = 0
        self.writer.add_scalar(tag, value, global_step=counter)

    def log(self, history: dict):
        logs = {
            key: value[-self.n_logs:] for key, value in history.items()
            }
        if self.clear_screen is True:
            clear()  # cleaning the screen up
        print(logs)


def get_logger(
        name: str,
        log_dir: Union[str, Path],
        n_logs: int,
        *args, **kwargs
        ) -> ILogger:
    if name in 'tb':
        return TBLogger(
            log_dir=log_dir,
            n_logs=n_logs,
            *args, **kwargs
        )
    raise NotImplementedError
