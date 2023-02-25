Tutorials
=========

Framework structure
-------------------

.. include:: structure.rst

-------------------------------------

Components
----------

.. include:: components.rst

-------------------------------------

Data Preperation
----------------

.. include:: data_formats.rst


-------------------------------------

Train your first model
----------------------

To start a training job in the SpeeQ framework, you need to configure three main
things: the model you want to train, the data you want to train the model on
and the processing pipeline for that data, and finally, the
training procedure. All of these can be done by building three configuration
objects, which can be found in the `speeq.config`` module:

- ModelConfig
- ASRDataConfig
- TrainerConfig

In the rest of this tutorial, we will explain how to create each of these objects
and how to successfully launch a training job.

Model Building
**************

.. include:: model_building.rst


Setting up data pipelines
*************************

.. include:: data_pipelines.rst


Training
********

.. include:: training.rst


Prediction
----------

.. include:: prediction.rst
