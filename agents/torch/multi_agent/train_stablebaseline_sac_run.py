"""Training baseline SAC with stablebaselines
"""

import os
from stable_baselines3 import SAC
from carla_env import CarlaEnv
from config import ENV_CONFIG


# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

ENV_CONFIG['num_agents'] = 1

env = CarlaEnv(ENV_CONFIG)

N_S = env.observation_space.shape[-1]
N_A = env.action_space.shape[-1]

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1600000, log_interval=5)
model.save("sac_stablebaseline")

