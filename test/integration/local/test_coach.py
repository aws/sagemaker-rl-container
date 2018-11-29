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


@pytest.mark.run_coach
def test_cartpole(docker_image, sagemaker_local_session, processor, tmpdir):
    source_dir = os.path.join(RESOURCE_PATH, 'coach_cartpole')
    dependencies = [os.path.join(RESOURCE_PATH, 'sagemaker_rl')]
    cartpole = 'train_coach.py'

    instance_type = 'local' if processor == 'cpu' else 'local_gpu'

    estimator = RLEstimator(entry_point=cartpole,
                            source_dir=source_dir,
                            role='SageMakerRole',
                            train_instance_count=1,
                            train_instance_type=instance_type,
                            sagemaker_session=sagemaker_local_session,
                            image_name=docker_image,
                            output_path='file://{}'.format(tmpdir),
                            dependencies=dependencies,
                            hyperparameters={
                                "save_model": 1,
                                "RLCOACH_PRESET": "preset_cartpole_clippedppo",
                                "rl.agent_params.algorithm.discount": 0.9,
                                "rl.evaluation_steps:EnvironmentEpisodes": 1,
                            })
    estimator.fit()

    local_mode_utils.assert_output_files_exist(str(tmpdir), 'output', ['success'])
    assert os.path.exists(os.path.join(str(tmpdir), 'model.tar.gz')), 'model file not found'
