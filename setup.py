# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from glob import glob
import os
from os.path import basename
from os.path import splitext

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_rl_container',
    version='1.0.0',
    description='Open source package for creating '
                'Reinforcement Learning containers to run on Amazon SageMaker.',
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.md'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-rl-container',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.6',
    ],

    install_requires=['numpy', 'tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock',
                      'sagemaker', 'docker-compose'],
)
