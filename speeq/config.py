from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from .interfaces import IProcessor, ITemplate


@dataclass
class TrainerConfig:
    """The trainer configuration

    Args:
        name (str): The trainer name.
        batch_size (int): the batch size.
        epochs (int): The number of training epochs.
        outdir (Union[Path, str]): The path to save the results to.
        logdir (Union[Path, str]): The path to save the logs to.
        log_steps_frequency (int): The number of steps to log
        the results after.
        criterion (str): The criterion name to be used.
        optimizer (str): The name of the optimizer to be used.
        optim_args (dict): The optimizer arguments.
        scheduler_template (Union[IScheduler, None]): The scheduler template to be used.
        Default None.
        dist_config (Union[object, None]): The DDP configuration object,
        for a single node/GPU training use None. Default None.
        logger (str): The logger name to be used. Default 'tb'.
        n_logs (int): The number of steps to log. Default 5.
        clear_screen (bool): whether to clear the screen after each log or
        not. Default False.
        criterion_args (dict): The criterion arguments if there is any.
        Default {}.
        grad_clip_thresh (Union[None, float]): max norm of the gradients.
        Default None.
        grad_clip_norm_type (float): type of the used p-norm. Default 2.0.
    """

    name: str
    batch_size: int
    epochs: int
    outdir: Union[Path, str]
    logdir: Union[Path, str]
    log_steps_frequency: int
    criterion: str
    optimizer: str
    optim_args: dict
    scheduler_template: Union[ITemplate, None] = None
    dist_config: Union[object, None] = None
    device: str = "cuda"
    logger: str = "tb"
    n_logs: int = 5
    clear_screen: bool = False
    criterion_args: dict = field(default_factory=dict)
    grad_clip_thresh: Union[None, float] = None
    grad_clip_norm_type: float = 2.0


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
        tokenizer_type (str): The tokenizer type to be used. Default
            `char_tokenizer`
        sep (str): the csv file's fields seprator. Default ','.
        type (str): the file type. Default 'csv'.
        padding_type (str): The padding to use static or dynamic.
        Default 'dynamic'.
        text_pad_max_len (str): Used if padding_type is static, which
        set the maximum sequence length it has to be larger than
        the largest text sequence size in both training and testing data.
        speech_pad_max_len (str): Used if padding_type is static, which
        set the maximum sequence length it has to be larger than
        the largest speech sequence size in both training and testing data.
        add_sos_token (bool): a flag if start of sentence token
            shall be added to the text squences. Default True.
        add_eos_token (bool): a flag if end of sentence token
            shall be added to the text squences. Default True.
        use_blank_as_pad (bool): A flag if the blank id to be used as padding.
            Default False.
        sort_key (Optional[str]): The key to sort the data on. Default ''.
        reverse (bool): Used if sorting key is passed, False will sort ascending,
            True will sort descending. Default is False.
        shuffle (bool): A flag indicating whether the dataset should be
            shuffled at each iteration Default False.
    """

    training_path: Union[str, Path]
    testing_path: Union[str, Path]
    speech_processor: IProcessor
    text_processor: IProcessor
    tokenizer_path: Union[str, Path]
    tokenizer_type: str = "char_tokenizer"
    sep: str = ","
    type: str = "csv"
    padding_type: str = "dynamic"
    text_pad_max_len: int = -1
    speech_pad_max_len: int = -1
    add_sos_token: bool = True
    add_eos_token: bool = True
    use_blank_as_pad: bool = False
    sort_key: str = ""
    reverse: bool = False
    shuffle: bool = False


@dataclass
class ModelConfig:
    """The model configuration.

    Args:
        template (ITemplate): The model template.
        model_path (Union[str, Path]): The pre-trained checkpoint
            to load the weights from. Default ''.
    """

    template: ITemplate
    model_path: Union[str, Path] = ""


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
