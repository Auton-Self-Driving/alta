"""Run Distributed Transfuser to collect offline data
"""

import os
import glob
import sys


CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

import carla

import os
import torch
import matplotlib.pyplot as plt

from network import PPOActorCritic_Continuous, PolicyNetwork, SoftQNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG, TEST_CONFIG


os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

# override config for testing
ENV_CONFIG.update(TEST_CONFIG)
ENV_CONFIG['initial_town'] = ENV_CONFIG['city_name']

env = CarlaEnv(ENV_CONFIG)
N_S = env.observation_space.shape[-1]
N_A = env.action_space.shape[-1]





env.close()

print('testing config:\n{}'.format(TEST_CONFIG))


