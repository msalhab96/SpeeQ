<div align="center">

[![Documentation Status](https://readthedocs.org/projects/speeq/badge/?version=latest)](https://speeq.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/msalhab96/SpeeQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/msalhab96/SpeeQ/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


 </div>

# SpeeQ
"SpeeQ", pronounced as "speekiu", is a Python-based speech recognition framework that allows developers and researchers to experiment and train various speech recognition models. It offers pre-implemented model architectures that can be trained with just a few lines of code, making it a suitable option for quick prototyping and testing of speech recognition models.

To get started, refer to the [documentation](https://speeq.readthedocs.io/en/latest/). If you need assistance or want to stay connected, please join our [Discord Server](https://discord.gg/Zfuyt7F3ZY).


# Installation

To install this package, you can follow the steps below:

1. Create and activate a Python environment using the following commands:


```bash
python3 -m venv env
source env/bin/activate
```

2. Install the packge either from source or from PyPI

  * from PyPI

    ```bash
    pip install speeq
    ```


  * from source

    ```bash
    git clone https://github.com/msalhab96/SpeeQ.git
    cd SpeeQ
    pip install .
    ```

# Implemented Models/Papers

| Model name      | Paper | Type |
| ---------------------- | ---------------------- | ---------------------- |
| Deep Speech 1 | [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) | CTC |
| Deep Speech 2 | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595) | CTC |
| Conformer | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) | CTC |
| Jasper | [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/abs/1904.03288) | CTC |
| Wav2Letter | [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/abs/1609.03193) | CTC |
| QuartzNet | [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions](https://arxiv.org/abs/1910.10261) | CTC |
| Squeezeformer | [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888) | CTC |
| RNNTransducer | [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711) | Transducer |
| ConformerTransducer | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) | Transducer |
| ContextNet | [ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context](https://arxiv.org/abs/2005.03191) | Transducer |
| VGGTransformer-Transducer | [Transformer-Transducer: End-to-End Speech Recognition with Self-Attention](https://arxiv.org/abs/1910.12977) | Transducer |
| Transformer-Transducer | [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562) | Transducer |
| BasicAttSeq2SeqRNN | N/A | Seq2Seq (encoder/decoder) |
| LAS | [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) | Seq2Seq (encoder/decoder) |
| RNNWithLocationAwareAtt | [Attention-Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503) | Seq2Seq (encoder/decoder) |
| SpeechTransformer | [Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](https://ieeexplore.ieee.org/document/8462506) | Seq2Seq (encoder/decoder) |


# Contributiuon
Your contributions are highly valued and appreciated! Our aim is to create an open and transparent environment that facilitates easy and straightforward contributions to this project. This can include reporting any issues or bugs you encounter, engaging in discussions regarding the current codebase, submitting fixes, proposing new features, or even becoming a maintainer yourself. We believe that your input is crucial to the continued growth and success of this framework. To start contributing to the framework, please consult the [guidelines](https://speeq.readthedocs.io/en/latest/files/contribution.html) for contributions.

# License & Citation
The framework is licensed under MIT. Therefore, if you use the framework, please consider citing it using the following bitex.

```
@software{Salhab_SpeeQ_A_framework_2023,
author = {Salhab, Mahmoud},
doi = {10.5281/zenodo.7708780},
license = {MIT},
month = {3},
title = {{SpeeQ: A framework for automatic speech recognition}},
url = {https://github.com/msalhab96/SpeeQ},
version = {0.0.1},
year = {2023}
}
```
