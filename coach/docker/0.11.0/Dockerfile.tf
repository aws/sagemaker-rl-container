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

# Install Redis.
RUN cd /tmp && \
    wget http://download.redis.io/redis-stable.tar.gz && \
    tar xvzf redis-stable.tar.gz && \
    cd redis-stable && \
    make && \
    make install

RUN pip install --no-cache-dir \
    PyOpenGL==3.1.0 \
    gym==0.10.5 \
    redis==2.10.6 \
    rl-coach-slim==0.11.0 && \
    pip install --no-cache-dir --upgrade sagemaker-containers

ENV COACH_BACKEND=tensorflow

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /
COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /opt/ml

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]
