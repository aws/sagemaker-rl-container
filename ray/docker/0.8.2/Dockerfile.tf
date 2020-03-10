ARG processor
ARG region
FROM 763104351884.dkr.ecr.$region.amazonaws.com/tensorflow-training:2.1.0-$processor-py36-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        ffmpeg \
        libjpeg-dev \
        libxrender1 \
        python3.6-dev \
        python3-opengl \
        wget \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    Cython==0.29.7 \
    tabulate \
    gym==0.12.1 \
    lz4 \
    opencv-python-headless==4.1.0.25 \
    PyOpenGL==3.1.0 \
    pyyaml \
    redis==3.3.2 \
    ray==0.8.2 \
    ray[rllib]==0.8.2 \
    scipy

# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /

COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]
