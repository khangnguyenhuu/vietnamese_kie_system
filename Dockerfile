FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ=Asia/Ho_Chi_Minh
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1 \
    CUDA_HOME="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Setup basic requirements
RUN apt-get update && apt-get upgrade -y
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    gnupg2 -y \
    libgl1-mesa-glx -y \
    git -y \
    python3 python3-pip zip \
    libatlas-base-dev python-dev python-ply python-numpy

WORKDIR /app

COPY . /app

# General package
RUN python3 -m pip install --upgrade pip && pip3 install torch==1.5.1 torchvision==0.6.1 \
    && pip3 install pyyaml==5.4.1 ninja yacs cython matplotlib tqdm opencv-python shapely scipy \
        tensorboardX pyclipper Polygon3 weighted-levenshtein editdistance easydict pythran \
    && pip3 install git+git://github.com/facebookresearch/detectron2.git@9eb4831f742ae6a13b8edb61d07b619392fb6543 \
    && pip3 install dict_trie nvidia-ml-py3
## MMDet package
RUN pip install mmdet

# MMOCR package
RUN cd app/libs/MMOCR &&\
    pip install -r requirements.txt &&\
    pip install -v -e .  # or "python setup.py develop" &&\
    export PYTHONPATH=$(pwd):$PYTHONPATH
# VietOCR package
RUN pip3 install vietocr

# Download model weights
RUN pip3 install gdown && bash download_weights.sh

# Setup for FastAPI
RUN pip3 install fastapi uvicorn[standard] python-multipart
ENV LANG=C.UTF-8

EXPOSE 80

CMD ["uvicorn", "main:app", "--workers", "1", "--host", "0.0.0.0", "--port", "80"]