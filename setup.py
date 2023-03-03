from setuptools import setup

from speeq._version import __version__

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.readlines()

with open("README.md", "r", encoding="utf-8") as f:
    description = f.read()
    
setup(
    name="speeq",
    version=__version__,
    author="Mahmoud Salhab",
    author_email="mahmoud@salhab.work",
    url="https://github.com/msalhab96/SpeeQ",
    install_requires=requirements,
    long_description = description,
    long_description_content_type='text/markdown',
    license="LICENSE",
    packages=["speeq"],
    setup_requires=["wheel"],
    python_requires=">=3.7",
    keywords=["speeq", "asr", "acoustic_modeling", "speech_recognition", "pytorch"],
    
)
