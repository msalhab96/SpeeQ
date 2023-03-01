To use the framework, it is essential to structure all datasets into a CSV file
that contains two fields: "file_path" for the audio file and "text" for the
transcription. If you want to sort the data, add a sorting column to the CSV
file. It's crucial to verify that all paths are valid and that the corresponding
files exist.

Here is an example of a CSV file

.. code-block:: bash

    file_path,text,cleaned_text,notes,duration
    audios/1.wav,the cat sat on the mat,the cat sat on the mat,A note!, 12.5
    audios/2.wav,what do you read?,what do you read,A note!, 3.5
