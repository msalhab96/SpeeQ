import random
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

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


class StochasticProcessor(OrderedProcessor):
    """Applies the provided processes in a stochastic way, where all processes
    first get shuffled then applied.

    Args:
        processes (List[IProcess]): The list of processes.
    """

    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__(processes)

    def execute(self, x):
        random.shuffle(self.processes)
        return super().execute(x)


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


class SpeechProcessor(IProcessor):
    """Speech processor the processing can be described as
    spec_augmenter(spec_processor(audio_augmenter(audio_processor(file_path))))
    if there's any feature extraction needed, it has to be part of the
    spectrogram processor

    Args:
        audio_processor (OrderedProcessor): The audio processor, where
            the input to it the audio file path.
        audio_augmenter (Optional[Union[OrderedProcessor, StochasticProcessor]]):
            The time domain augmentation processor. Default None.
        spec_processor (Optional[OrderedProcessor]): The spectrogram processor,
            where the input is the signal in the time domain, if any feature
            extraction is needed it has to be part of it. Default None.
        spec_augmenter (Optional[Union[OrderedProcessor, StochasticProcessor]]):
            The frequency domain augmentation process. default None.
    """

    def __init__(
        self,
        audio_processor: OrderedProcessor,
        audio_augmenter: Optional[Union[OrderedProcessor, StochasticProcessor]] = None,
        spec_processor: Optional[OrderedProcessor] = None,
        spec_augmenter: Optional[Union[OrderedProcessor, StochasticProcessor]] = None,
    ) -> None:
        super().__init__()
        self.processors = []
        self.__add(audio_processor)
        self.__add(audio_augmenter)
        self.__add(spec_processor)
        self.__add(spec_augmenter)
        if spec_augmenter is not None:
            assert spec_processor is not None

    def __add(
        self, processor: Union[OrderedProcessor, StochasticProcessor, None]
    ) -> None:
        if processor is not None:
            self.processors.append(processor)

    def execute(self, file_path: Union[str, Path]):
        x = file_path
        for processor in self.processors:
            x = processor.execute(x)
        return x
