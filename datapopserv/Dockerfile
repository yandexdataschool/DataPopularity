FROM ubuntu:14.04
MAINTAINER Mikhail Hushchyn <mikhail91@yandex-team.ru>

#install packages for rep
RUN apt-get update && apt-get install -y \
  python2.7 \
  python-pip \
  python-dev \
  libatlas-base-dev \
  gfortran \
  liblapack-dev

RUN apt-get update && apt-get install git dpkg-dev make g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev libpng3 libjpeg8 gfortran libssl-dev libpcre3-dev libgl1-mesa-dev libglew1.5-dev libftgl-dev libmysqlclient-dev libfftw3-dev libcfitsio3-dev graphviz-dev libavahi-compat-libdnssd-dev libldap2-dev python-dev libxml2-dev libkrb5-dev libgsl0-dev libqt4-dev curl python-pycurl --force-yes -y

RUN apt-get install libcurl4-gnutls-dev -y --force-yes

RUN easy_install distribute
RUN pip install cython
RUN pip install "ipython[notebook]"
RUN pip install \
  numpy==1.9.1 \
  pandas==0.14.0

RUN pip install \
  pyzmq==14.3.0 \
  pycurl==7.19.3 \
  scipy==0.14.0 \
  plotly \
  numexpr==2.4 \
  jinja2==2.7.3

RUN pip install scikit-learn==0.15.2
RUN pip install matplotlib==1.3.1

#install rep
RUN pip install -i http://pypi.yandex-team.ru/simple/ rep

RUN sudo apt-get install -y git

#install DataPopularity
RUN git clone https://github.com/hushchyn-mikhail/DataPopularity.git
RUN pip install DataPopularity/Packages/kernel_regression-master/.
RUN pip install DataPopularity/.

#install packages for data_popularity_api
RUN pip install flask
RUN pip install flask-restful
