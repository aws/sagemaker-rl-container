ARG processor
FROM 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.11.0-$processor-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        libav-tools \
        libjpeg-dev \
        libxrender1 \
        python3.6-dev \
        python3-opengl \
        wget \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    gym==0.10.5 \
    lz4 \
    PyOpenGL==3.1.0 \
    opencv-python \
    redis==2.10.6 \
    ray==0.5.3 \
    ray[rllib]==0.5.3 \
    scipy && \
    pip install --no-cache-dir --upgrade sagemaker-containers

# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /

COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]
