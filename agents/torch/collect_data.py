import os
import json
import datetime
import argparse

import numpy as np
import cv2

from carla_env import CarlaEnv
from agents.navigation.behavior_agent import BehaviorAgent


def collect_trajectory(env, save_dir, behavior='cautious', max_path_length=5000):
    now = datetime.datetime.now()
    salt = np.random.randint(100)
    fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
    save_path = os.path.join(save_dir, fname)
    rgb_path = os.path.join(save_path, 'rgb')
    topdown_path = os.path.join(save_path, 'topdown')
    measurements_path = os.path.join(save_path, 'measurements')

    # check for conflicts
    if os.path.isdir(save_path):
        print('Directory conflict, trying again...')
        return 0
    
    # make directories
    os.mkdir(save_path)
    os.mkdir(rgb_path)
    os.mkdir(topdown_path)
    os.mkdir(measurements_path)

    seed = np.random.randint(10000000)
    obs = env.reset(n_vehicles=0, n_pedestrians=0, seed=seed)
    agent = env._agent

    for step in range(max_path_length):
        control, target_speed = agent.run_step()

        if abs(obs[0]) > .075:
            target_speed *= .5

        action = np.array([control.steer, target_speed])
        next_obs, reward, done, info = env.step(action)

        # save state
        rgb = info.pop('rgb')
        topdown = info.pop('topdown')

        del info['walker']
        del info['vehicle']

        experience = {
            'obs': obs.tolist(),
            'next_obs': next_obs.tolist(),
            'action': action.tolist(),
            'reward': reward,
            'done': done
        }
        experience.update(info)

        save_env_state(rgb, topdown, experience, save_path, step)

        if done:
            break

        obs = next_obs

    return step + 1


def save_env_state(rgb, topdown, measurements, save_path, idx):
    rgb_path = os.path.join(save_path, 'rgb', '{:04d}.png'.format(idx))
    cv2.imwrite(rgb_path, rgb)

    topdown_path = os.path.join(save_path, 'topdown', '{:04d}.png'.format(idx))
    cv2.imwrite(topdown_path, topdown)

    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def main(args):
    with CarlaEnv(carla_gpu=args.gpu, town=args.town) as env:
        total_samples = 0
        while total_samples < args.n_samples:
            traj_length = collect_trajectory(env, args.path, args.behavior)
            total_samples += traj_length

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--behavior', type=str, default='cautious')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
