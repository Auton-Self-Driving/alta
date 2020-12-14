"""A3C agent using discrete_actions
"""

import os
import queue
import datetime
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

from a3c_utils import SharedAdam, push_and_pull, record, v_wrap
from a3c_network import Basic_Discrete
from a3c_env import CarlaEnv
from a3c_env_config import ENV_CONFIG

from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent



class _A3C_Individual_Agent(Agent):
    def __init__(self, vehicle, glb_net=None, rank=None,
        device='cuda:0', **kwargs):
        """A local indivial A3C agent.
        Args:
            vehicle: ego-vehicle in env (e.g. env.vehicle_actor)
            glb_net: network shared by all A3C agents, not required for MP
            rank: an integer for identification of this local agent, 
                not required for MP
            device: the device for local and global net
            **kwargs: include proximity_threshold=10.0, 
                traffic_light_proximity_threshold=10.0, 
                vehicle_proximity_threshold=15.0
        """
        super().__init__(vehicle, **kwargs)
        self.glb_net = glb_net
        if self.glb_net is not None:
             self.local_net = pickle.loads(pickle.dumps(self.glb_net))
             self.local_net = self.local_net.to(device)
             self.buffer_s = []
             self.buffer_a = []
             self.buffer_r = []
        self.rank = rank
        self.device = device
        self.done = False
        self.action = None
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        self.vehicle_actor = vehicle
        self.num_total_steps = 0
        self.episode_reward = 0
        self.curr_reward = 0
        self.observation = None
    
    def run_step(self, obs):
        if self.glb_net is None: raise NotImplementedError(
            'Not intended for MP A3C Agent, \
            Call A3C_MP_Agent.local_net.choose_action(obs) instead')
        return self.local_net.choose_action(obs)

    def update_local_net(self):
        if self.glb_net is None: raise NotImplementedError(
            'Not intended for MP A3C Agent')
        self.local_net.load_state_dict(self.glb_net.state_dict())

    def update_buffer(self, s, a, r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def reset_buffer(self):
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []


class A3C_Collective_Agent(object):
    def __init__(self, glb_env, glb_net, glb_optimizer, num_agents=1, 
        max_glb_num_episodes=10000, glb_update_freq=5, device='cpu'):
        """An torch.multiprocessing A3C agent.
        Args:
            glb_env: the global environment
            glb_net: network shared by all A3C agents
            glb_optimizer: optimizer for the global_net
            num_agents: number of A3C agents
            max_glb_num_episodes: max number of global episodes
            glb_update_freq: update frequency of glb_net
        """
        super().__init__()
        self.glb_env = glb_env
        self.glb_net = glb_net
        self.glb_optimizer = glb_optimizer
        self.max_glb_num_episodes = max_glb_num_episodes
        self.glb_update_freq = glb_update_freq
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.res_queue = [[] for _ in self.rank_list ]
        self.agent_list = None
        self.device = device

    def to_tensor(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def _push_and_pull(self, rank, done, s_, gamma=.9):
        if self.agent_list is None:
            raise ValueError('Should run self.learn() first')
        agent = self.agent_list[rank]
        bs, ba, br = agent.buffer_s, agent.buffer_a, agent.buffer_r

        v_s_ = 0 if done else agent.local_net.forward(
            self.to_tensor(s_[None, :]))[-1].detach().cpu().numpy()[0, 0]

        buffer_v_target = []
        for r in br[::-1]: # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        loss = agent.local_net.loss_func(
            self.to_tensor(np.vstack(bs)),
            self.to_tensor(np.array(ba), dtype=np.int64) if ba[0].dtype == \
                np.int64 else self.to_tensor(np.vstack(ba)),
            self.to_tensor(np.array(buffer_v_target)[:, None]))

        # calculate local gradients and push local parameters to global
        self.glb_optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(agent.local_net.parameters(), 
            self.glb_net.parameters()):
            gp._grad = lp.grad
        self.glb_optimizer.step()

        # pull global parameters
        agent.update_local_net()
        # empty buffer
        agent.reset_buffer()

    def learn(self):
        glb_num_episodes = 0
        # initialize
        # obs_list = self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.spawn_npc_vehicles()
        self.agent_list = [_A3C_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_net=self.glb_net, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()
        while glb_num_episodes < self.max_glb_num_episodes:
            for rk, agent in enumerate(self.agent_list):
                prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                action = agent.local_net.choose_action(
                    prev_obs.to(self.device))
                agent.action = action
            # step forward
            print('action chosen:', [agt.action for agt in self.agent_list])
            self.glb_env.step()

            for rk, agent in enumerate(self.agent_list):
                agent.update_buffer(prev_obs, agent.action, agent.curr_reward)

                if agent.num_total_steps % self.glb_update_freq == 0 or \
                    agent.done:  
                    # update global and assign to local net
                    self._push_and_pull(rk, agent.done, agent.observation)

                if agent.done:  # done and print information
                    print('[Agent {}] done, episode reward [{}]'.format(
                        rk, agent.episode_reward))
                    glb_num_episodes += 1

                agent.num_total_steps += 1
            # respawn agent
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list):
                if agent.done: respawn_rank_list.append(rk)
            if len(respawn_rank_list) > 0: # there're dead agents to respawn
                self.glb_env.reset(rank_list=respawn_rank_list)
                # update agent list
                for rk in respawn_rank_list:
                    self.agent_list[rk] = _A3C_Individual_Agent(
                        self.glb_env.ego_vehicle_list[rk],
                        glb_net=self.glb_net, rank=rk)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()

    def run(self):
        raise NotImplementedError('This agent does not use MP')


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
                action = self.local_net.choose_action(obs)
                new_obs, reward, done, ep_info = self.glb_env.step(action)
                done = bool(done[0, 0])

                if done: reward = ep_info['total_reward']
                ep_r += reward

                buffer_s.append(obs)
                buffer_a.append(action)
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

    '''
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
    '''

    print('*' * 80)
    print('FINISHED')
    print('*' * 80)




