FROM python:3.8

RUN apt-get update -y && apt-get upgrade -y

RUN pip install --upgrade pip

COPY 'requirements.txt' .

COPY 'dev_requirements.txt' .

RUN pip install -r requirements.txt

RUN pip install -r dev_requirements.txt

RUN pip install jupyter
