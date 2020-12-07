"""A3C agent using discrete_actions
"""

import os
import pickle
import torch
import torch.multiprocessing as mp

from a3c_utils import SharedAdam, push_and_pull, record
from a3c_network import Basic_Discrete
from a3c_env import CarlaEnv
from a3c_env_config import ENV_CONFIG

from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent

class _A3C_Individual_Agent(Agent):
    def __init__(self, vehicle, **kwargs):
        """A local indivial A3C agent.
        Args:
            vehicle: ego-vehicle in env (e.g. env.vehicle_actor)
            **kwargs: include proximity_threshold=10.0, 
                traffic_light_proximity_threshold=10.0, 
                vehicle_proximity_threshold=15.0
        """
        super().__init__(vehicle, **kwargs)
    
    def run_step(self):
        raise NotImplementedError(
            'Call A3C_MP_Agent.glb_net.choose_action(obs) instead')


class A3C_MP_Agent(mp.Process):
    def __init__(self, glb_env, glb_net, glb_optimizer, 
        glb_num_episodes, glb_episode_reward, glb_queue, 
        name='N/A', max_glb_num_episodes=10000, glb_update_freq=5):
        """An torch.multiprocessing A3C agent.
        Args:
            glb_env: the global environment
            glb_net: network shared by all A3C agents
            glb_optimizer: optimizer for the global_net
            glb_num_episodes: global number of episodes
            glb_episode_reward: global rewards
            glb_queue: global multiprocessing queue
            name: a string for identification of this local agent
            max_glb_num_episodes: max number of global episodes
            glb_update_freq: update frequency of glb_net
        """
        super().__init__()
        self.name = '[WorkerID {}]'.format(name)
        # self.glb_env = CarlaEnv(ENV_CONFIG)
        self.glb_env = glb_env
        self.glb_net = glb_net
        self.glb_optimizer = glb_optimizer
        self.glb_queue = glb_queue
        self.glb_num_episodes = glb_num_episodes
        self.glb_episode_reward = glb_episode_reward
        self.max_glb_num_episodes = max_glb_num_episodes
        self.glb_update_freq = glb_update_freq
        self.local_net = pickle.loads(pickle.dumps(self.glb_net)) # local net

    def run(self):
        print('[61]', flush=True)
        total_step = 1
        while self.glb_num_episodes.value < self.max_glb_num_episodes:
            print('[64]', flush=True)
            obs = self.glb_env.reset()
            print('[66]', flush=True)
            _ = _A3C_Individual_Agent(self.glb_env.vehicle_actor)
            print('[67]', flush=True)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                # print('[72]', flush=True)
                obs = torch.from_numpy(obs).to(torch.float)
                control = self.local_net.choose_action(obs)
                new_obs, reward, done, ep_info = self.glb_env.step(control)
                done = bool(done[0, 0])

                if done: reward = ep_info['total_reward']
                ep_r += reward

                buffer_s.append(obs)
                buffer_a.append(control)
                buffer_r.append(reward)

                obs = new_obs

                if total_step % self.glb_update_freq == 0 or done:  
                    # update global and assign to local net
                    # sync
                    push_and_pull(self.glb_optimizer, self.local_net, self.glb_net, 
                        done, new_obs, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        # record(self.glb_num_episodes, self.glb_episode_reward,
                            # ep_r, self.glb_queue, self.name)
                        obs = self.glb_env.reset()
                        _ = _A3C_Individual_Agent(self.glb_env.vehicle_actor)
                        break

                total_step += 1

        self.glb_queue.put(None)
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["OMP_NUM_THREADS"] = '1'
    # mp.set_sharing_strategy('file_system')
    # mp.set_start_method('spawn')

    env = CarlaEnv(ENV_CONFIG)

    N_S = env.observation_space.shape[-1]
    N_A = env.action_space.n
    print(N_S, N_A)
    N_S = 22
    N_A = 12
    # from IPython import embed; embed()

    glb_net = Basic_Discrete(N_S, N_A) # global network
    glb_net.share_memory() # share the global parameters in multiprocessing
    glb_optimizer = SharedAdam(glb_net.parameters(), lr=1e-4, betas=(0.92, 0.999))
    glb_num_episodes = mp.Value('i', 0)
    glb_episode_reward = mp.Value('d', 0)
    glb_queue = mp.Queue()

    # parallel training
    workers = [A3C_MP_Agent(env, glb_net, glb_optimizer, glb_num_episodes, 
        glb_episode_reward, glb_queue, name='FOO') for i in range(1)]
    [w.start() for w in workers]
    # [w.run() for w in workers]
    res = [] # record episode reward to plot
    while True:
        print('122 122 122')
        r = glb_queue.get()
        print('127 127 127')
        print(r)
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    print('*' * 80)
    print('FINISHED')
    print('*' * 80)




