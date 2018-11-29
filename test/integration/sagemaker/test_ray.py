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
from timeout import timeout


@pytest.mark.run_ray
def test_ray_tf(sagemaker_session, ecr_image, instance_type):
    source_dir = os.path.join(RESOURCE_PATH, 'ray_cartpole')
    cartpole = 'train_ray.py'

    estimator = RLEstimator(entry_point=cartpole,
                            source_dir=source_dir,
                            role='SageMakerRole',
                            train_instance_count=1,
                            train_instance_type=instance_type,
                            sagemaker_session=sagemaker_session,
                            image_name=ecr_image)

    with timeout(minutes=15):
        estimator.fit()
