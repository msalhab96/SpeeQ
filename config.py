from pathlib import Path
from typing import Union
from utils.loggers import ILogger
from dataclasses import dataclass, field
from data.interfaces import IProcessor
from models.templates import ITemplate


@dataclass
class TrainerConfig:
    """The trainer configuration

    Args:
        batch_size (int): the batch size.
        epochs (int): The number of training epochs.
        outdir (Union[Path, str]): The path to save the results to.
        logger (ILogger): The logger to be used.
        log_steps_frequency (int): The number of steps to log
        the results after.
        optimizer (str): The name of the optimizer to be used.
        optim_args (dict): The optimizer arguments.
        schedular (Union[str, None]): The name of the schedular to be used.
        Default None.
        schedular_args (dict): The schedular arguments. Default {}.
        dist_config (Union[object, None]): The DDP configuration object,
        for a single node/GPU training use None. Default None.
    """
    batch_size: int
    epochs: int
    outdir: Union[Path, str]
    logger: ILogger
    log_steps_frequency: int
    optimizer: str
    optim_args: dict
    schedular: Union[str, None] = None
    schedular_args: dict = field(default_factory=dict)
    dist_config: Union[object, None] = None
    device: str = 'cuda'


@dataclass
class ASRDataConfig:
    """The ASR data configuration

    Args:
        training_path (Union[str, Path]): The training data path.
        testing_path (Union[str, Path]): The testing data path.
        speech_processor (IProcessor): The speech processor.
        text_processor (IProcessor): The text processor.
        tokenizer_path (Union[str, Path]): The path to load or save
        the tokenizer.
    """
    training_path: Union[str, Path]
    testing_path: Union[str, Path]
    speech_processor: IProcessor
    text_processor: IProcessor
    tokenizer_path: Union[str, Path]
    sep: str = ','
    type: str = 'csv'
    padding_type: str = 'dynamic'
    pad_max_len: int = -1


@dataclass
class ModelConfig:
    """The model configuration.

    Args:
        template (ITemplate): The model template.
        model_path (Union[str, Path, None]): The pre-trained checkpoint
        to load the weights from. Default None.
    """
    template: ITemplate
    model_path: Union[str, Path, None] = ''


@dataclass
class DistConfig:
    """The distributed data parallel configuration.

    Args:
        port (int): The port used.
        n_gpus (int): The number of nodes/GPUs.
        address (str): The master node address.
        backend (str): The backend to be used for DDP.
    """
    port: int
    n_gpus: int
    address: str
    backend: str
