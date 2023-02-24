from setuptools import find_packages, setup

with open(".version", "r", encoding="utf-8") as f:
    version = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.readlines()

setup(
    name="speeq",
    version=version,
    author="Mahmoud Salhab",
    author_email="mahmoud@salhab.work",
    url="https://github.com/msalhab96/SpeeQ",
    install_requires=requirements,
    license="LICENSE",
    packages=["speeq"],
    setup_requires=["wheel"],
)
