The function `speeq.trainers.trainers.launch_training_job` is used to start a training job, and it takes three input parameters:
the model configuration, the data configuration, and the training configuration. The following code demonstrates how
to create the trainer configuration, which specifies the training settings:


.. code-block:: python

    from speeq.config import TrainerConfig, DistConfig

    # create a single-GPU training configuration
    signle_gpu_trainer_cfg = TrainerConfig(
        name='seq2seq', # can be 'ctc', 'seq2seq', or 'transducer', depending on the selected model
        batch_size=16,
        epochs=100,
        outdir='outdir',
        logdir='outdir/logs',
        log_steps_frequency=100,
        criterion='ctc',
        optimizer='adam',
        optim_args={'lr': 0.005},
        device='cuda'
    )

    # use DDP for multiple-GPU training
    dist_cfg = DistConfig(
        port=12345,
        n_gpus=2,
        address='tcp://localhost:5555',
        backend='nccl'
    )
    multiple_gpus_trainer_cfg = TrainerConfig(
        name='seq2seq',
        batch_size=16,
        epochs=100,
        outdir='outdir',
        logdir='outdir/logs',
        log_steps_frequency=100,
        criterion='ctc',
        optimizer='adam',
        optim_args={'lr': 0.005},
        dist_config=dist_cfg
    )

After creating the trainer configuration, and having the model and data configuration
objects, you can easily launch the training job using the following code:


.. code-block:: python

    from speeq.trainers.trainers import launch_training_job

    launch_training_job(
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config
        )
