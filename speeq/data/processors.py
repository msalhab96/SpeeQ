""""This module includes implementations of classes that fulfill the abstract class IProcessor and define the execute method as an interface. The following classes are available:

- OrderedProcessor: applies a series of processes in a specific order.
- StochasticProcessor: applies a sequence of processes in a randomized order.
- SpeechProcessor: a higher-level class that wraps a sequence of processors used for speech processing.

Usage:

The classes in this module are designed to be used together to process speech signals. The SpeechProcessor class provides a high-level interface to the processing pipeline, while the OrderedProcessor and StochasticProcessor classes can be used to construct custom processing pipelines. The classes can be used as follows:

1. OrderedProcessor:

This class applies a sequence of processes in order.

Example:

    .. code-block:: python

        from speeq.interfaces import IProcess
        from speeq.data.processors import OrderedProcessor

        class MyProcess1(IProcess):
            def run(self, x: Any) -> Any:
                # Process x here
                return x

        class MyProcess2(IProcess):
            def run(self, x: Any) -> Any:
                # Process x here
                return x

        processes = [MyProcess1(), MyProcess2()]
        processor = OrderedProcessor(processes)
        output = processor.execute(input_data)


2. StochasticProcessor:

This class applies a sequence of processes in a randomized order.

    .. code-block:: python

        from speeq.interfaces import IProcess
        from speeq.data.processors import StochasticProcessor

        class MyProcess1(IProcess):
            def run(self, x: Any) -> Any:
                # Process x here
                return x

        class MyProcess2(IProcess):
            def run(self, x: Any) -> Any:
                # Process x here
                return x

        processes = [MyProcess1(), MyProcess2()]
        processor = StochasticProcessor(processes)
        output = processor.execute(input_data)

    .. note::

        All of the classes in this module inherit from the IProcessor abstract
        class and implement the execute method. This allows them to be used
        interchangeably in processing pipelines.

"""

import random
from pathlib import Path
from typing import Any, List, Optional, Union

from speeq.interfaces import IProcess, IProcessor


class OrderedProcessor(IProcessor):
    """Applies a list of provided processes in a specific order. The order of the
    processes is determined by their position in the list.

    Args:
        processes (List[IProcess]): A list of IProcess objects representing the processes
        to be applied in order.

    Example:

        .. code-block:: python

            # Import required packages
            from speeq.data.processors OrderedProcessor
            from speeq.data.processes import AudioLoader, FeatExtractor

            input_data = 'path/to/file.wav'

            # Define a list of processes
            processes = [
                AudioLoader(sample_rate=16000),
                FeatExtractor(feat_ext_name='mfcc', feat_ext_args={'n_mfcc': 10})
            ]

            # Create an instance of the OrderedProcessor class
            processor = OrderedProcessor(processes=processes)

            # Apply the list of processes in order to some input data
            processed_data = processor.execute(input_data)

    """

    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__()
        self.processes = processes

    def execute(self, x: Any) -> Any:
        """Executes all processes on the input x in the order they were provided.
        The output of the previous process is used as the input for the next process.

        Args:
            x (Any): The input

        Returns:
            Any: The output data after applying all the processes in order.
        """
        for process in self.processes:
            x = process.run(x)
        return x


class StochasticProcessor(OrderedProcessor):
    """Applies the provided processes in a stochastic order. The order in which
    the processes are applied is randomized for each input, making this class
    suitable for data augmentation.

    Args:
        processes (List[IProcess]): A list of processes to be applied in a stochastic order.
    """

    def __init__(self, processes: List[IProcess]) -> None:
        super().__init__(processes)

    def execute(self, x: Any) -> Any:
        """Executes all the processes on the input x in a randomly shuffled order.

        Args:
            x (Any): The input to be processed.

        Returns:
            Any: The output of the processed input.
        """
        random.shuffle(self.processes)
        return super().execute(x)


class SpeechProcessor(IProcessor):
    """Speech processor that applies a series of processing steps to audio data.
    The processing steps can be described as:
    spec_augmenter(spec_processor(audio_augmenter(audio_processor(file_path))))

    .. note::

        If feature extraction is needed, it has to be part of the spectrogram processor.


    Args:
        audio_processor (OrderedProcessor): The audio processor that takes the
        audio file path as input.

        audio_augmenter (Optional[Union[OrderedProcessor, StochasticProcessor]]):
        The time-domain augmentation processor. Defaults to None.

        spec_processor (Optional[OrderedProcessor]): The spectrogram processor
        that takes the signal in the time domain as input. If any feature
        extraction is needed, it has to be part of this processor. Defaults to None.

        spec_augmenter (Optional[Union[OrderedProcessor, StochasticProcessor]]): The
        frequency-domain augmentation processor. Defaults to None.
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
