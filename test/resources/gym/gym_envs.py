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
import argparse
import gym
from gym import wrappers
import os


def run_simulation(env_name, output_dir):
    print('*' * 86)
    print('Running {} simulation.'.format(env_name))

    env_dir = os.path.join(output_dir, env_name)
    print('Saving results to \'{}\'.'.format(env_dir))
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    env = gym.make(env_name)
    # record every episode
    # (by default video recorder only captures a sampling of episodes:
    # those with episodes numbers which are perfect cubes: 1, 8, 27, 64, ... and then every 1000th)
    env = wrappers.Monitor(env, env_dir, force=True, video_callable=lambda episode_id: True)
    for i_episode in range(3):
        print('Start Episode #' + str(i_episode) + '-' * 86)
        env.reset()
        total_reward = 0
        for step in range(1000):
            env.render(mode='rgb_array')
            action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Env: {}, Episode: {},\tSteps: {},\tReward: {}".format(
                    env_name, i_episode, step, total_reward))
                break
                env.reset()
        print('End' + '-' * 86)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Algorithms - 'Reverse-v0'
    # Atari - 'Breakout-ram-v0', 'Breakout-v0'
    # Box2D - 'LunarLander-v2'
    # Classic control - 'MountainCar-v0'
    # Toy text - 'FrozenLake-v0'
    parser.add_argument('--envs', type=list, default=[
        'Reverse-v0',
        'Breakout-ram-v0',
        'Breakout-v0',
        'LunarLander-v2',
        'MountainCar-v0',
        'CartPole-v0',
        'FrozenLake-v0'
    ])

    args, unknown = parser.parse_known_args()

    print(args)

    for env_name in args.envs:
        run_simulation(env_name, '/opt/ml/output/intermediate')

    with open('/opt/ml/model/model.txt', mode='w') as f:
        f.write('Success!')
