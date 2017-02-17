FROM ubuntu:14.04

RUN sudo apt-get -y update
RUN sudo apt-get install -y python-dev libblas-dev libatlas-dev liblapack-dev gfortran g++ python-setuptools
RUN sudo apt-get install -y libpng-dev libjpeg8-dev libfreetype6-dev libxft-dev
RUN which pip || sudo easy_install pip
RUN sudo pip install --upgrade pip

# Git Installation
RUN sudo apt-get -y update
RUN sudo apt-get install -y git

# Clone repository
WORKDIR home/
RUN git clone https://github.com/yandexdataschool/DataPopularity.git
WORKDIR DataPopularity/
RUN git pull
RUN git checkout develop
WORKDIR /home/

# Datapop installation
RUN sudo pip install DataPopularity/datapop

# Datapopserv installation
RUN sudo pip install DataPopularity/datapopserv

# Run service preparation
RUN mkdir workdir
ENV WORKING_DIR=/home/workdir
WORKDIR DataPopularity/datapopserv/datapopserv
