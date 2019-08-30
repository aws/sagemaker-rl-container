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

import sagemaker
from sagemaker.rl import RLEstimator
from test.integration import RESOURCE_PATH
import local_mode_utils


def test_vw_serving(local_instance_type, sagemaker_local_session, docker_image, tmpdir,
                                        training_data_bandits, pretrained_model_vw, role):
                                        
    environ_vars = {"AWS_DEFAULT_REGION": "us-west-2",
                    "EXPERIMENT_ID": "test-exp-1",
                    "EXP_METADATA_DYNAMO_TABLE": "test",
                    "MODEL_METADATA_DYNAMO_TABLE": "test",
                    "AWS_REGION": "us-west-2",
                    "FIREHOSE_STREAM": "none",
                    "LOG_INFERENCE_DATA": "false",
                    "MODEL_METADATA_POLLING": "false"}
    
    model = sagemaker.model.Model(
        image=docker_image,
        role=role,
        env=environ_vars,
        name="test_env",
        model_data=pretrained_model_vw
        )
    model.deploy(initial_instance_count=1, instance_type=local_instance_type)

    predictor = sagemaker.predictor.RealTimePredictor(
        model.endpoint_name,
        serializer=sagemaker.predictor.json_serializer,
        deserializer=sagemaker.predictor.json_deserializer,
        sagemaker_session=model.sagemaker_session)
    
    resp = predictor.predict({"observation": [1,3], "request_type": "observation"})
    print(resp)

    for key in ["action", "action_prob", "event_id", "timestamp", "sample_prob", "model_id"]:
        assert key in resp
    
    predictor.delete_endpoint()
    