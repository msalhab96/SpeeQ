Within the framework, models can be constructed using what is referred to as a
"**template**." This template serves as a structure that must be filled with
the model configuration, and all available model templates can be found in
the "speeq.models.templates" module.

There are two types of model templates:

- **Static templates**: These allow you to alter only the model configuration while keeping the model architecture unchanged. This option provides the flexibility to replicate previous research works.
- **Dynamic templates**: These permit changes to both the model architecture and configuration, enabling the combination of components from different layers or the creation of custom models for that reason we call them `model builders`


Within the "speeq.models.templates" module, only three Dynamic templates
(model builders) are available. These are:

- CTCModelBuilderTemp: This template is utilized for constructing and customizing CTC models.
- Seq2SeqBuilderTemp: This template is used for constructing and customizing Seq2Seq models (Encoder/Decoder).
- TransducerBuilderTemp: This template is used for constructing and customizing Transducer models.

All other templates within the module are Static templates.


Example on Static templates:

let's assume you want to experiment with Speech transformer model to build its
template, you can easily do the below, and it is the same for any model's architecture.

.. code-block:: python

    # importing the templates module
    from torch.nn import Softmax
    from speeq.models import templates

    template = template.SpeechTransformerTemp(
        in_features=160,
        n_conv_layers=3,
        kernel_size=32,
        stride=2,
        d_model=512,
        n_enc_layers=8,
        n_dec_layers=8,
        ff_size=1024,
        h=8,
        att_kernel_size=16,
        att_out_channels=512,
        pred_activation=nn.Softmax(dim=-1)
    )


Example on Dynamic templates:

let's assume you want to experiment with a dummy feed-forward based CTC model to build its
template, you can easily do the below.


.. code-block:: python

    # importing the templates module
    from torch import nn
    from speeq.models import templates

    # define the encoder
    class Encoder(nn.Module):
        def __init__(self, in_features: int, feat_size: int):
            super().__init__()
            self.fc = nn.Linear(in_features, feat_size)

        def forward(self, x, mask):
            # x os shape [B, M, in_features]
            lengths = mask.sum(dim=-1)
            out = self.fc(x)
            return out, lengths

    # define an instance of the encoder
    feat_size = 512
    encoder = Encoder(80, feat_size)

    # define the template
    template = templates.CTCModelBuilderTemp(
        encoder=encoder,
        feat_size=feat_size
    )


Once you have defined the model structure and architecture using template, it is the time to create
the model. This can be accomplished by creating a configuration object that
includes the template and the path of a pre-trained model, if available. You can
then pass the model configuration object to the `get_model` method found in the
`speeq.models.registry` module, as shown in the code below.

.. code-block:: python

    # importing registry to use the get_model funciton
    from speeq.models import registry, templates
    # import ModelConfig to setup model configuration
    from speeq.config import ModelConfig

    # defining a dummy template
    template = template.SpeechTransformerTemp(
        in_features=160,
        n_conv_layers=3,
        kernel_size=32,
        stride=2,
        d_model=512,
        n_enc_layers=8,
        n_dec_layers=8,
        ff_size=1024,
        h=8,
        att_kernel_size=16,
        att_out_channels=512,
        pred_activation=nn.Softmax(dim=-1)
    )

    # creating model configuration object
    model_cfg = ModelConfig(template=template)
    # creating the model
    model = registry.get_model(model_config=model_cfg, n_classes=5)
