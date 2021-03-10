FROM ubuntu:16.04

RUN apt-get -y remove "^python*"

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libboost-all-dev \
        zlib1g-dev \
        cmake \
        git \
        curl \
        nginx \
        wget \
        python3.6-dev && ln -s -f /usr/bin/python3.6 /usr/bin/python;

RUN cd /usr/lib/x86_64-linux-gnu/ && ln -s libboost_python-py35.so libboost_python3.so

# Install pip
RUN cd /tmp && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && rm get-pip.py

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

# Build VW
RUN git clone --branch 8.7.0.post1_with_python https://github.com/VowpalWabbit/vowpal_wabbit.git && \
    cd vowpal_wabbit && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DVW_INSTALL=ON -DBUILD_PYTHON=ON -DPY_VERSION=3.6 && \
    make install

# For VW python bindings
ENV PYTHONPATH /vowpal_wabbit/python:/vowpal_wabbit/build/python

# Install Redis.
RUN \
  cd /tmp && \
  wget http://download.redis.io/redis-stable.tar.gz && \
  tar xvzf redis-stable.tar.gz && \
  cd redis-stable && \
  make && \
  make install

RUN pip install ipython \
                sagemaker-containers==2.5.1 \
                redis==3.2.1 \
                jsonschema==3.0.1 \
                retrying==1.3.3 \
                smart-open==1.8.4 \
                pandas==0.23


# Install vw-serving
ADD src/vw-serving /opt/code/vw-serving
RUN pip install /opt/code/vw-serving

WORKDIR /opt

