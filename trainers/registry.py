import os
from data.registry import get_asr_loaders, get_tokenizer
from interfaces import ITrainer
from models.registry import get_model
from torch.optim import (
    Adam, RMSprop, SGD
)
from trainers.criterions import CTCLoss
from trainers.trainers import CTCTrainer, DistCTCTrainer
from utils.loggers import get_logger
from utils.utils import set_state_dict

CRITERIONS = {
    'ctc': CTCLoss
}

OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'sgd': SGD
}

TRAINERS = {
    'ctc': CTCTrainer
}

DIST_TRAINERS = {
    'ctc': DistCTCTrainer
}


def get_criterion(
        name: str,
        blank_id: int,
        pad_id: int,
        *args,
        **kwargs
        ):
    assert name in CRITERIONS
    return CRITERIONS[name](
        blank_id=blank_id,
        pad_id=pad_id,
        *args,
        **kwargs
    )


def get_optimizer(model, trainer_config):
    return OPTIMIZERS[trainer_config.optimizer](
        model.parameters(), **trainer_config.optim_args
        )


def _get_asr_trainer_args(
        rank: int,
        world_size: int,
        trainer_config,
        data_config,
        model_config
        ) -> dict:
    logger = get_logger(
        name=trainer_config.logger,
        log_dir=trainer_config.logdir,
        n_logs=trainer_config.n_logs,
        clear_screen=trainer_config.clear_screen
        )
    tokenizer = get_tokenizer(
        data_config=data_config
        )
    model = get_model(
        model_config=model_config,
        n_classes=tokenizer.vocab_size
        )
    optimizer = get_optimizer(
        model=model, trainer_config=trainer_config
    )
    criterion = get_criterion(
        name=trainer_config.criterion,
        blank_id=tokenizer.special_tokens.blank_id,
        pad_id=tokenizer.special_tokens.pad_id,
        **trainer_config.criterion_args
    )
    if os.path.exists(model_config.model_path):
        *_, history = set_state_dict(
            model=model,
            optimizer=optimizer,
            state_path=model_config.model_path
            )
    else:
        history = {}
    train_loader, test_loader = get_asr_loaders(
        data_config=data_config,
        tokenizer=tokenizer,
        batch_size=trainer_config.batch_size,
        world_size=world_size,
        rank=rank
    )
    args = {
        'optimizer': optimizer,
        'criterion': criterion,
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'epochs': trainer_config.epochs,
        'log_steps_frequency': trainer_config.log_steps_frequency,
        'logger': logger,
        'history': history,
    }
    if world_size == 1:
        args['device'] = trainer_config.device
    return args


def _get_dist_args(
        trainer_config,
        rank: int,
        world_size: int
        ) -> dict:
    return {
        'rank': rank,
        'world_size': world_size,
        'dist_address': trainer_config.dist_config.address,
        'dist_port': trainer_config.dist_config.port,
        'dist_backend': trainer_config.dist_config.backend
    }


def get_asr_trainer(
        rank: int,
        world_size: int,
        trainer_config,
        data_config,
        model_config
        ) -> ITrainer:
    name = trainer_config.name
    base_args = _get_asr_trainer_args(
        rank=rank, world_size=world_size,
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config
    )
    args = dict(
        **base_args if world_size == 1 else dict(
            **base_args, **_get_dist_args(
                rank=rank, world_size=world_size,
                trainer_config=trainer_config
                )
            )
        )
    if world_size == 1:
        return TRAINERS[name](**args)
    return DIST_TRAINERS[name](**args)
