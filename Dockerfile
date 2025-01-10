FROM ic-registry.epfl.ch/ivrl/pajouheshgar/pytorch2.01:cuda11.7v2

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    vim \
    git \
    wget \
    curl \
    && apt-get clean

RUN python3 -m pip install --upgrade pip \
    && pip install torch torchvision \
    matplotlib numpy tqdm \
    transformers accelerate \
    scipy scikit-learn pandas trimesh objaverse \
    wandb datasets iopath \
    && pip install diffusers==0.31.0 \
    && pip install --no-deps --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html \
    && pip install --upgrade jupyterlab

CMD ["/bin/bash"]
