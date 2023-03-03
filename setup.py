from setuptools import setup

from speeq._version import __version__

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.readlines()

setup(
    name="speeq",
    version=__version__,
    author="Mahmoud Salhab",
    author_email="mahmoud@salhab.work",
    url="https://github.com/msalhab96/SpeeQ",
    install_requires=requirements,
    license="LICENSE",
    packages=["speeq"],
    setup_requires=["wheel"],
    python_requires=">=3.7",
)
