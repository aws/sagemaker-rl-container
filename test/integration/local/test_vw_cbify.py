# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import pytest

from sagemaker.rl import RLEstimator
from test.integration import RESOURCE_PATH
import local_mode_utils


def test_vw_cb_explore(local_instance_type, sagemaker_local_session, docker_image,
                       tmpdir, training_data_supervised, role):
    source_path = os.path.join(RESOURCE_PATH, 'vw')
    estimator = RLEstimator(entry_point="train_cbify.py",
                            source_dir=source_path,
                            role=role,
                            instance_count=1,
                            instance_type=local_instance_type,
                            hyperparameters={"num_arms": 7},
                            sagemaker_session=sagemaker_local_session,
                            output_path='file://{}'.format(tmpdir),
                            image_uri=docker_image)
    estimator.fit(inputs=training_data_supervised)

    local_mode_utils.assert_output_files_exist(str(tmpdir), 'output', ['success'])
    local_mode_utils.assert_output_files_exist(str(tmpdir), 'model', ['vw.model'])    
    assert os.path.exists(os.path.join(str(tmpdir), 'model.tar.gz')), 'model file not found'