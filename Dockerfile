FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video
ENV TORCH_CUDA_ARCH_LIST=Turing

#RUN git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization && \
 RUN python3 -m pip install ./diff-gaussian-rasterization && \
    # simple-knn
    python3 -m pip install ./simple-knn && \
    # nvdiffrast
 #   pip install git+https://github.com/NVlabs/nvdiffrast/ && 
    python3 -m pip install ./nvdiffrast && \
  # kiuikit
    python3 -m pip install git+https://github.com/ashawkey/kiuikit
   # pip install ./kiuikit
#RUN --mount=type=cache,target=/root/.cache/pip \
  #  --mount=type=bind,source=requirements.txt,target=requirements.txt \
  #  python -m pip install -r requirements.txt
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3 python3-dev python3-pip python3-setuptools python3-distutils && \
    apt install -y git && \
    apt install ffmpeg libsm6 libxext6 libglm-dev -y && \
    apt clean && rm -rf /var/lib/apt/lists/* &&

COPY requirements.txt /requirements.txt
COPY . .


RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install Cmake &&\
    python3 -m pip install --no-cache-dir -r /requirements.txt &&\
    python3 -m pip install -q --upgrade pip &&\
    python3 -m pip install prophet  &&\
    python3 -m pip install pyproject-toml
# Switch to the non-privileged user to run the application.
#USER appuser

# Copy the source code into the container.
#COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD uvicorn 'main:app' --host=0.0.0.0 --port=8000
