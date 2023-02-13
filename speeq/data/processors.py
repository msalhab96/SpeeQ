import random
from abc import abstractmethod
from typing import List

from speeq.interfaces import IProcess, IProcessor


class OrderedProcessor(IProcessor):
    """Applies all the provided processes in order.

    Args:
        processes (List[IProcess]): The list of processes.
    """

    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__()
        self.processes = processes

    def execute(self, x):
        for process in self.processes:
            x = process.run(x)
        return x


class StochasticProcessor(IProcessor):
    """Applies the provided processes in a stochastic way, where all processes
    first get shuffled then applied.

    Args:
        processes (List[IProcess]): The list of processes.
    """

    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__(processes)

    def execute(self, x):
        random.shuffle(self.processes)
        super().execute(x)


class StochasticProcess(IProcess):
    """Applies the process functionality based on the ratio provided

    Args:
        ratio (float): The rate of applying the process on the input.
    """

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
