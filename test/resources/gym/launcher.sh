#!/usr/bin/env bash

pip install gym[atari] gym[box2d] box2d-py>=2.3.5
python gym_envs.py "$@"
