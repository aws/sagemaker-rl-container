ARG processor
ARG region
FROM 520713654638.dkr.ecr.$region.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-$processor-py3

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

# Update awscli for compatibility with the latest botocore version that breaks it
# https://github.com/boto/boto3/issues/2596
RUN pip install --upgrade awscli

RUN pip install --no-cache-dir \
    PyOpenGL==3.1.0 \
    pyglet==1.3.2 \
    gym==0.12.5 \
    redis==2.10.6 \
    rl-coach-slim==1.0.0 && \
    pip install --no-cache-dir --upgrade sagemaker-containers && \
    pip install --upgrade numpy

ENV COACH_BACKEND=tensorflow

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /
COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

WORKDIR /opt/ml

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]
