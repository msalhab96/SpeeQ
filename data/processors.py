import random
from abc import abstractmethod
from typing import List

from interfaces import IProcess, IProcessor


class OrderedProcessor(IProcessor):
    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__()
        self.processes = processes

    def execute(self, x):
        for process in self.processes:
            x = process.run(x)
        return x


class StochasticProcessor(IProcessor):
    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__(processes)

    def execute(self, x):
        random.shuffle(self.processes)
        super().execute(x)


class StochasticProcess(IProcess):
    def __init__(self, ratio: float) -> None:
        super().__init__()
        self.ratio = ratio

    @property
    def _shall_do(self) -> bool:
        return random.random() <= self.ratio

    @abstractmethod
    def func():
        pass

    def run(self, x):
        if self._shall_do:
            return self.func(x)
        return x
