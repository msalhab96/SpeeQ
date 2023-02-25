The main components of the SpeeQ framework are:

Config
******
Includes all configuration objects that are necessary to initialize and instantiate
various objects and jobs, such as trainers, predictors, models, and data input/output.


Data
****
Encompasses all data-related components, including data augmenters, data loaders,
data padders, and tokenizers, as well as various data preprocessing pipelines.

Models
******
Comprises all pre-implemented speech recognition models, along with layers, encoders, and decoders.

Predictors
**********
Consists of different modules for speech recognition prediction, which can be
used in the inference stage from a pre-trained models.

Trainers
********
incorporates all modules and components required for training speech recognition models.

Utils
*****
contains various helper functionalities and training loggers.
