import sys, os, time, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
import gym
import numpy as np
import time
import traceback
import csv

from datetime import datetime
import tensorboard_logging as tf_log

# PPO specific
import baselines.common.tf_util as U

# NOTE: not using baselines logger for now
# from baselines import logger
from baselines import deepq
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

if __name__ == '__main__':
    with U.make_session():
        # Create the environment
       
        # logger = tf_log.Logger(TB_LOGS_DIR)
        env = gym.make('CartPole-v0')
        
        env.reset()
        obs = env.render(mode='rgb_array')
        print('SEMANTIC SHAPE: ', obs.shape)
        import ipdb; ipdb.set_trace()
