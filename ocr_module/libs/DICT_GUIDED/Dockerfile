FROM continuumio/miniconda:latest
COPY . /DICT_GUIDED/
WORKDIR /DICT_GUIDED/
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    vim wget screen -y \
    gnupg2 -y
RUN apt-get update \
    && conda init bash \
    && . ~/.bashrc \
    && conda create -n dict-guided -y python=3.7 \
    && conda activate dict-guided \
    && conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch \
    && apt update \
    && apt install -y build-essential \
    && apt-get install manpages-dev \
    && python -m pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance \
    && apt-get update \
    && apt install -y libgl1-mesa-glx \
    && python -m pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html \
    && pip install dict_trie \
    && python setup.py install
