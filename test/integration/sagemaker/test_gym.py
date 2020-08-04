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

from sagemaker.rl import RLEstimator
from test.integration import RESOURCE_PATH
from timeout import timeout


def test_gym(sagemaker_session, ecr_image, instance_type, framework):
    resource_path = os.path.join(RESOURCE_PATH, 'gym')
    gym_script = 'launcher.sh' if framework == 'tensorflow' else 'gym_envs.py'
    estimator = RLEstimator(entry_point=gym_script,
                            source_dir=resource_path,
                            role='SageMakerRole',
                            instance_count=1,
                            instance_type=instance_type,
                            sagemaker_session=sagemaker_session,
                            image_uri=ecr_image)

    with timeout(minutes=15):
        estimator.fit()
