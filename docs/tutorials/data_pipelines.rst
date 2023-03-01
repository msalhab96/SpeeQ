Having the data in the CSV format as mentioned earlier, we can use now the data loaders
that built in the framework, in-order to train a model or launch a training job
we have to define a speech and text processor first and after that the `speeq.config.ASRDataConfig` object.

Defining Speech and Text processor
**********************************
Both processors are requird to be an instance of `speeq.data.processors.OrderedProcessor`,
which contains a list of instances from process objects that implement the
`speeq.interfaces.IProcess` interface. For convenience, the framework provides
a set of predefined processes that are commonly used, and they can be found in
the `speeq.data.processes` module.


Text Processor
++++++++++++++

Suppose we want to create a dummy text processor that consists of two
processes: text stripping (which removes trailing and leading white spaces
from the input text) and lowering the text. We can do this by creating the
processes first and then passing them to the processor, as shown in the code below:


.. code-block:: python

    from speeq.data.processors import OrderedProcessor
    from speeq.interfaces import IProcess

    # Create the text stripper process.
    class TextStripper(IProcess):
        def run(self, text: str) -> str:
            return text.strip()

    # Create the text lowering process.
    class TextLowering(IProcess):
        def run(self, text: str) -> str:
            return text.lower()

    # Create a list of the processes that will be executed by the processor in order.
    processes = [TextStripper(), TextLowering()]

    # Create the text processor.
    text_processor = OrderedProcessor(processes)


Speech Processor
++++++++++++++++
The speech processor can be created in a similar way as the text processor. However,
for the speech processor, there are predefined processes that fall into two
categories: speech processors and speech augmenters. These can be accessed from
the `speeq.data.processes` and `speeq.data.augmenters` modules.

The processes must always be executed in a specific order, whereas the
augmenters can be applied in various orders to produce different augmentation
combinations. To address this, we have two types of processors: `OrderedProcessor`
and `StochasticProcessor`. The former applies processes in sequential order,
while the latter shuffles the processes before applying them sequentially to the input.


To combine these processors, we have the speeq.data.processors.SpeechProcessor module. Below is an
example of how to create a speech processor in both scenarios:


.. code-block:: python

    from speeq.data.processors import SpeechProcessor, OrderedProcessor, StochasticProcessor
    from speeq.data.processes import FeatExtractor
    from speeq.data.augmenters import FrequencyMasking, WhiteNoiseInjector

    # Create a speech processor that loads speech and extracts mel-scale spectrogram
    sample_rate = 16000
    speech_processor = OrderedProcessor([
        AudioLoader(sample_rate=sample_rate),
        FeatExtractor(feat_ext_name='melspec', feat_ext_args={})
    ])

    # If you want to add data augmentation in an ordered process, you can add augmenters as follows:
    speech_processor_with_aug = OrderedProcessor([
        AudioLoader(sample_rate=sample_rate),
        # Time-domain augmentation
        WhiteNoiseInjector(ratio=0.3),
        FeatExtractor(feat_ext_name='melspec', feat_ext_args={}),
        # Frequency-domain augmentation
        FrequencyMasking(n=5, max_length=10, ratio=0.2)
    ])

    # However, if you have more than one time or frequency domain and you want to shuffle their execution order,
    # you can use the `SpeechProcessor` module to achieve that as follows:

    speech_file_processor = OrderedProcessor([
        AudioLoader(sample_rate=sample_rate)
    ])
    time_domain_aug = StochasticProcessor([
        WhiteNoiseInjector(ratio=0.3),
        VariableAttenuator(ratio=0.4)
    ])
    spec_processor = OrderedProcessor([
        FeatExtractor(feat_ext_name='melspec', feat_ext_args={})
    ])
    freq_domain_aug = StochasticProcessor([
        FrequencyMasking(n=5, max_length=10, ratio=0.2)
    ])
    speech_processor_with_rand_aug = SpeechProcessor(
        audio_processor=speech_file_processor,
        audio_augmenter=time_domain_aug,
        spec_processor=spec_processor,
        spec_augmenter=freq_domain_aug
    )

    """The speech_processor_with_rand_aug will perform the following steps in a
    specific order: first, it will provide the file path to speech_file_processor,
    then it will pass the time domain signal to the time domain augmentation. After
    that, it will extract the features using spec_processor and, finally, apply frequency
    domain augmentation using freq_domain_aug."""

Building ASRDataConfig
**********************



Once the text processor and speech processor are built, we can create the data configuration
object, which is similar to the model configuration. The code below demonstrates how to create an
ASRDataConfig object:


.. code-block:: python

    from speeq.config import ASRDataConfig

    data_cfg = ASRDataConfig(
        training_path='path/to/train.csv',
        testing_path='path/to/test.csv',
        speech_processor=speech_processor,
        text_processor=text_processor,
        tokenizer_path='outdir/tokenizer.json',
        tokenizer_type='char_tokenizer',
        add_sos_token=True,
        add_eos_token=True,
        sort_key='duration'
    )



This will create a configuration object for ASR with training and testing data paths,
speech and text processors, tokenizer information, and sorting criteria.
