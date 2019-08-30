#  Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import os
from setuptools import setup, find_packages
from os.path import basename
from os.path import splitext
from glob import glob


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='vw_serving',
    version=read('VERSION').strip(),
    description='Open source library for serving/hosting VW based Bandits/RL Algorithms on Amazon SageMaker.',
    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    entry_points={
        "console_scripts":
        [
            "serve=vw_serving.model_manager:main",
        ]
    },
)
