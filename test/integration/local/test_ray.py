#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest

from sagemaker.rl import RLEstimator
from test.integration import RESOURCE_PATH
import local_mode_utils


@pytest.mark.run_ray
def test_ray(local_instance_type, sagemaker_local_session, docker_image, tmpdir, framework):
    source_dir = os.path.join(RESOURCE_PATH, 'ray_cartpole')
    cartpole = 'train_ray_tf.py' if framework == 'tensorflow' else 'train_ray_torch.py'

    estimator = RLEstimator(entry_point=cartpole,
                            source_dir=source_dir,
                            role='SageMakerRole',
                            instance_count=1,
                            instance_type=local_instance_type,
                            sagemaker_session=sagemaker_local_session,
                            output_path='file://{}'.format(tmpdir),
                            image_uri=docker_image)

    estimator.fit()

    local_mode_utils.assert_output_files_exist(str(tmpdir), 'output', ['success'])
    assert os.path.exists(os.path.join(str(tmpdir), 'model.tar.gz')), 'model file not found'


