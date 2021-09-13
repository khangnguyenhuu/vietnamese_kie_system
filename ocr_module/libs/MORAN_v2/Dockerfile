FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

COPY . /moran/
WORKDIR /moran/

RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    vim wget screen -y \
    gnupg2 -y

RUN apt-get update \
    && apt-get install python3.8 -y \
    && apt-get update \
    && apt-get install python3-pip -y \
    && python3 -m pip install --upgrade pip \
    && apt-get update \
    && apt install -y libgl1-mesa-glx \
    && pip3 install -r requirements.txt
