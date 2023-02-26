To utilize a pre-trained model for prediction, you can follow these steps. First,
create a speech processor object and a model configuration object as done previously.
However, you need to provide the path to the model configuration object to load the pre-trained model.
Finally, you can use a predictor from the speeq.predictors module.

Here is an example of how to use a CTC predictor with a pre-trained
model and a speech processor object:

.. code-block:: python

    from speeq.predictors import CTCPredictor

    predictor = CTCPredictor(
        speech_processor=speech_processor,
        tokenizer_path='path/to/tokenizer.json',
        model_config=model_config,
        device='cuda',
    )

    # Use the predictor to transcribe an audio file
    print(predictor.predict('path/to/audio.wav'))
