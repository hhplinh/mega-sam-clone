FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_HOME=/usr/local/cuda

ENV TORCH_DONT_CHECK_COMPILER_VERSION=1

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
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12

# Set GCC-12 as the default compiler (Priority 100)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# ---- Project ----
# 1. COPY THE CODE FIRST
COPY base /tmp/base
WORKDIR /tmp/base

#----------------

RUN rm -rf thirdparty/eigen && \
    wget -qO - https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz && \
    mv eigen-3.4.0 thirdparty/eigen

# RUN find src thirdparty -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/\.scalar_type()/.scalar_type()/g' {} +

ENV TORCH_DONT_CHECK_COMPILER_VERSION=1
ENV MAX_JOBS=1
ENV TORCH_NVCC_FLAGS="-w"
ENV CFLAGS="-w"
ENV CXXFLAGS="-w"
#----------------------

RUN python setup.py install

RUN pip install viser

WORKDIR /mega_sam
ENV PYTHONPATH="/mega_sam/UniDepth:$PYTHONPATH"

ENV MKL_THREADING_LAYER=GNU

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc
RUN echo "conda activate mega_sam" >> /root/.bashrc