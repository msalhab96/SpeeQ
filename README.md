# SpeeQ
"SpeeQ", pronounced as "speekiu", is a Python-based speech recognition framework that allows developers and researchers to experiment and train various speech recognition models. It offers pre-implemented model architectures that can be trained with just a few lines of code, making it a suitable option for quick prototyping and testing of speech recognition models.

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
