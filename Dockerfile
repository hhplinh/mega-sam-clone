FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# FROM nvcr.io/nvidia/pytorch:25.01-py3


ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"

ENV TORCH_DONT_CHECK_COMPILER_VERSION=1

ENV TORCH_DONT_CHECK_COMPILER_VERSION=1
ENV MAX_JOBS=1
ENV TORCH_NVCC_FLAGS="-w"
ENV CFLAGS="-w"
ENV CXXFLAGS="-w"
ENV USE_NINJA=1

# 2. Add CUDA binaries to PATH
ENV PATH=${CUDA_HOME}/bin:${PATH}

# 3. Add CUDA libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    ffmpeg \
    tar \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniconda ----
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# ---- Conda env ----
COPY environment.yml /tmp/environment.yml
WORKDIR /tmp

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "mega_sam", "/bin/bash", "-c"]

# ---- Torch stack ----
# RUN python -m pip install --upgrade pip setuptools wheel
# RUN pip install torch==2.8.0.dev20250626+cu128 torchvision==0.23.0.dev20250627+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

# RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install --no-build-isolation --pre -v -U git+https://github.com/facebookresearch/xformers.git@fde5a2fb46e3f83d73e2974a4d12caf526a4203e

# RUN pip install nvidia-cusparselt-cu12
RUN pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git

RUN pip install ninja tyro plotly
# RUN pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12

# Set GCC-12 as the default compiler (Priority 100)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100


COPY viser /tmp/viser
WORKDIR /tmp
# RUN echo "Searching for 'viser' folder..." && \
#     find . -iname "viser" -type d

RUN pip install -e viser

# ---- Project ----
# 1. COPY THE CODE FIRST
COPY base /tmp/base
WORKDIR /tmp/base

#----------------

RUN rm -rf thirdparty/eigen && \
    wget -qO - https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz && \
    mv eigen-3.4.0 thirdparty/eigen

#----------------------

# 1. Clear any pre-existing build artifacts that might have been copied over
RUN rm -rf build/ dist/ *.egg-info


# 2. Re-compile specifically for Blackwell (SM 12.0)
# Use 'conda run' to ensure the environment's nvcc and python are used
# RUN conda run -n mega_sam python setup.py clean --all && \
#     TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA="1" \
#     conda run -n mega_sam python setup.py install

RUN TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA="1" \
    conda run -n mega_sam python setup.py install

WORKDIR /mega_sam
ENV PYTHONPATH="/mega_sam/UniDepth:$PYTHONPATH"


ENV MKL_THREADING_LAYER=GNU

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc
RUN echo "conda activate mega_sam" >> /root/.bashrc