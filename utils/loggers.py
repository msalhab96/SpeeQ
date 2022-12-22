import pandas as pd
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
    def __init__(self, log_dir: Union[str, Path], n_logs: int) -> None:
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self.__counters = dict()
        self.n_logs = n_logs
        clear()
        print('Started!')

    def log_step(
            self,
            key: str,
            category: str,
            value: Union[int, float]
            ) -> None:
        tag = f'{key/category}'
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
        logs = pd.DataFrame(logs)
        clear()  # cleaning the screen up
        print(logs)
