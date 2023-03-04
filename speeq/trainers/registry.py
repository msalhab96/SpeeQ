"""A factory for creating speech task trainers.

This module provides functions to create various objects needed for training
speech models, such as loss functions, optimizers, and trainers. The functions
are implemented as factory methods, which allow for abstracting object creation
and facilitate the creation of customized trainers.

Functions:
    get_criterion(name: str, blank_id: int, pad_id: int) -> torch.nn.Module:
        Returns a PyTorch module that computes the loss for a speech recognition
        task. The `name` argument specifies the type of loss to use, and the
        `blank_id` and `pad_id` arguments are used to configure the loss
        function.

    get_optimizer(model: torch.nn.Module, trainer_config) -> Union[torch.optim.Optimizer, IScheduler]:
        Returns a PyTorch optimizer or learning rate scheduler for training a
        speech model. The `model` argument is the PyTorch module to be trained,
        and the `trainer_config` argument is a configuration object containing
        the hyperparameters for training.

    get_trainer(trainer_config, data_config, model_config, rank=0, world_size=1) -> ITrainer:
        Returns a speech task trainer object. The `trainer_config` argument is a
        configuration object containing the hyperparameters for training, the
        `data_config` argument is a configuration object containing the
        parameters the training data, the `model_config` argument
        is a configuration object containing the parameters for building the
        speech model, and the `rank` and `world_size` arguments are used for
        distributed training.
"""

import os
from typing import Union

from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from speeq.config import ASRDataConfig, ModelConfig, TrainerConfig
from speeq.data.registry import get_asr_loaders, get_tokenizer
from speeq.interfaces import IScheduler, ITrainer
from speeq.models.registry import get_model
from speeq.utils.loggers import get_logger
from speeq.utils.utils import get_text_list, load_csv, set_state_dict

from .criterions import CrossEntropyLoss, CTCLoss, NLLLoss, RNNTLoss
from .schedulers import NoamScheduler, SqueezeformerNoamScheduler
from .trainers import (
    CTCTrainer,
    DistCTCTrainer,
    DistSeq2SeqTrainer,
    DistTransducerTrainer,
    Seq2SeqTrainer,
    TransducerTrainer,
)

CRITERIONS = {
    "ctc": CTCLoss,
    "crossentropy": CrossEntropyLoss,
    "nllloss": NLLLoss,
    "rnnt": RNNTLoss,
}

OPTIMIZERS = {"adam": Adam, "adamw": AdamW, "rmsprop": RMSprop, "sgd": SGD}

TRAINERS = {
    "ctc": CTCTrainer,
    "seq2seq": Seq2SeqTrainer,
    "transducer": TransducerTrainer,
}

DIST_TRAINERS = {
    "ctc": DistCTCTrainer,
    "seq2seq": DistSeq2SeqTrainer,
    "transducer": DistTransducerTrainer,
}

SCHEDULERS = {"noam": NoamScheduler, "squeezeformer_noam": SqueezeformerNoamScheduler}


def get_criterion(name: str, blank_id: int, pad_id: int, *args, **kwargs):
    """This function generates and returns a module representing a criterion.

    Args:

        name (str): The name of the criterion.

        blank_id (int): The ID for the blank symbol used in the criterion.

        pad_id (int): The ID for the padding symbol used in the criterion.

    Returns:

        Module: The desired criterion module.

    """
    assert name in CRITERIONS
    return CRITERIONS[name](blank_id=blank_id, pad_id=pad_id, *args, **kwargs)


def get_optimizer(model, trainer_config) -> Union[Optimizer, IScheduler]:
    """This function generates and provides an optimizer or scheduler, based on
    the input model and training configuration.

    Args:

        model (Module): The model.

        trainer_config (object): The configuration object for training.

    Returns:

        Union[Optimizer, IScheduler]: The optimizer or scheduler object that
        will be used for training.

    """
    if trainer_config.scheduler_template is not None:
        return SCHEDULERS[trainer_config.scheduler_template.name](
            params=model.parameters(),
            optimizer=trainer_config.optimizer,
            optimizer_args=trainer_config.optim_args,
            **trainer_config.scheduler_template.get_dict()
        )
    return OPTIMIZERS[trainer_config.optimizer](
        model.parameters(), **trainer_config.optim_args
    )


def _get_asr_trainer_args(
    rank: int,
    world_size: int,
    trainer_config: TrainerConfig,
    data_config: ASRDataConfig,
    model_config: ModelConfig,
) -> dict:
    logger = get_logger(
        name=trainer_config.logger,
        log_dir=trainer_config.logdir,
        n_logs=trainer_config.n_logs,
        clear_screen=trainer_config.clear_screen,
    )
    data = load_csv(data_config.training_path, sep=data_config.sep)
    data = get_text_list(data=data)
    tokenizer = get_tokenizer(data_config=data_config, data=data)
    model = get_model(model_config=model_config, n_classes=tokenizer.vocab_size)
    if world_size == 1:
        model = model.to(trainer_config.device)
    optimizer = get_optimizer(model=model, trainer_config=trainer_config)
    criterion = get_criterion(
        name=trainer_config.criterion,
        blank_id=tokenizer.special_tokens.blank_id,
        pad_id=tokenizer.special_tokens.pad_id,
        **trainer_config.criterion_args
    )
    if os.path.exists(model_config.model_path):
        ignore = trainer_config.ignore_optim_state
        *_, history = set_state_dict(
            model=model,
            optimizer=optimizer if ignore is False else None,
            state_path=model_config.model_path,
        )
    else:
        history = {}
    train_loader, test_loader = get_asr_loaders(
        data_config=data_config,
        tokenizer=tokenizer,
        batch_size=trainer_config.batch_size,
        world_size=world_size,
        rank=rank,
    )
    args = {
        "optimizer": optimizer,
        "criterion": criterion,
        "model": model,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "epochs": trainer_config.epochs,
        "log_steps_frequency": trainer_config.log_steps_frequency,
        "logger": logger,
        "outdir": trainer_config.outdir,
        "history": history,
    }
    if world_size == 1:
        args["device"] = trainer_config.device
    return args


def _get_dist_args(trainer_config, rank: int, world_size: int) -> dict:
    return {
        "rank": rank,
        "world_size": world_size,
        "dist_address": trainer_config.dist_config.address,
        "dist_port": trainer_config.dist_config.port,
        "dist_backend": trainer_config.dist_config.backend,
    }


def get_asr_trainer(
    trainer_config: TrainerConfig,
    data_config: ASRDataConfig,
    model_config: ModelConfig,
    rank: int = 0,
    world_size: int = 1,
) -> ITrainer:
    """Creates an ASR trainer object for training a speech recognition model.

    Args:

        trainer_config (TrainerConfig): A configuration object that specifies
        settings for the trainer.

        data_config (ASRDataConfig): A configuration object that specifies
        settings for the data used in training.

        model_config (ModelConfig): A configuration object that specifies
        settings for the model architecture.

        rank (int, optional): The rank of the current process, for distributed
        training. Defaults to 0.

        world_size (int, optional): The number of processes for distributed
        training. Defaults to 1.

    Returns:
        ITrainer: An object that encapsulates the ASR trainer functionality.
    """
    name = trainer_config.name
    base_args = _get_asr_trainer_args(
        rank=rank,
        world_size=world_size,
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config,
    )
    args = dict(
        **base_args
        if world_size == 1
        else dict(
            **base_args,
            **_get_dist_args(
                rank=rank, world_size=world_size, trainer_config=trainer_config
            )
        )
    )
    if world_size == 1:
        return TRAINERS[name](**args)
    return DIST_TRAINERS[name](**args)
