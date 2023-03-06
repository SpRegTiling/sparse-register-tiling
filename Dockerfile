FROM ubuntu:jammy
ARG DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

#
# Dev Container
#

RUN apt-get update && apt-get install -y \
        build-essential             \
        software-properties-common  \
        sudo wget unzip htop cmake git


# Install MKL
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    add-apt-repository -y "deb https://apt.repos.intel.com/oneapi all main" && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-get update -y && \
    apt-get install -y intel-oneapi-mkl-devel-2021.4.0

# Install PAPI
RUN apt-get -y install libpapi-dev

# Install Python3.8
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3.8-distutils python3-pip python3-apt

ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m pip install virtualenv
RUN python3.8 -m virtualenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install PyTorch
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install other python dependencies 
RUN pip install pandas  \
		seaborn \
		matplotlib \
		altair altair_saver \
		scipy \
		tabulate \
		pyyaml

RUN wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip; \
    gunzip /usr/local/bin/ninja.gz; \
    chmod a+x /usr/local/bin/ninja

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections; \
    apt-get install -y ttf-mscorefonts-installer

ENV DATASET_DIR=/datasets