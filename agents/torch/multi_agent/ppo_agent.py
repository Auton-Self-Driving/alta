import torch
import torch.nn as nn
import gym
import numpy as np

from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG
from environment.carla_9_4.agents.navigation.agent import Agent

class _PPO_Individual_Agent(Agent):
    def __init__(self, vehicle, local_policy, glb_policy, rank=None,
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
        self.local_policy = local_policy
        self.glb_policy = glb_policy
        self.reset_memory()
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

    def select_action(self):
        prev_state = self.observation
        state_tensor = torch.from_numpy(prev_state).to(torch.float)
        action, logprob = self.local_policy.act(state_tensor)
        # update partial memory
        self.memory['state'].append(prev_state)
        self.memory['action'].append(action)
        self.memory['logprob'].append(logprob)
        return action

    def update_local_policy(self):
        self.local_policy.load_state_dict(self.global_policy.state_dict())

    def reset_memory(self):
        self.memory = {
            'action': [],
            'state': [],
            'logprob': [],
            'reward': [],
            'done': [],}


class PPO_Collective_Agent(object):
    def __init__(self, glb_env, glb_policy, glb_optimizer, num_agents=1,
        max_glb_num_episodes=10000, glb_update_freq=5, verbose=False):
        """An torch.multiprocessing PPO agent.
        Args:
            glb_env: the global environment
            glb_policy: network shared by all A3C agents
            glb_optimizer: optimizer for the global_net
            num_agents: number of A3C agents
            max_glb_num_episodes: max number of global episodes
            glb_update_freq: update frequency of glb_policy
            verbose: if print some debug information
        """
        super().__init__()
        self.glb_env = glb_env
        self.glb_policy = glb_policy
        self.glb_optimizer = glb_optimizer
        self.max_glb_num_episodes = max_glb_num_episodes
        self.glb_update_freq = glb_update_freq
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        self.device = next(glb_policy.parameters()).device
        self.verbose = verbose
        self.glb_ep_reward_list = []
        self.agent_reward_list = [[] for _ in self.rank_list]

    def vprint(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

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
            self.glb_policy.parameters()):
            gp._grad = lp.grad
        self.glb_optimizer.step()

        # pull global parameters
        agent.update_local_net()
        # empty buffer
        agent.reset_buffer()

    def learn(self):
        glb_num_episodes = 1
        # initialize
        # obs_list = self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.spawn_npc_vehicles()
        self.agent_list = [_PPO_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()

        avg_t_action, avg_t_step  = [], []

        while glb_num_episodes < self.max_glb_num_episodes + 1:
            # take action
            ts_action = time.time()
            for rk, agent in enumerate(self.agent_list):
                prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                action = agent.local_net.choose_action(
                    prev_obs.to(self.device))
                agent.action = action
            te_action = time.time()
            self.vprint('action chosen:', [a.action for a in self.agent_list])
            # get new observation
            ts_step = time.time()
            self.glb_env.step()
            te_step = time.time()
            avg_t_action.append(te_action - ts_action)
            avg_t_step.append(te_step - ts_step)
            self.vprint('[num_agent {}][action time {:.4f}, avg {:.4f}]'
                '[step time {:.4f}, avg {:.4f}]'.format(self.num_agents,
                avg_t_action[-1], np.mean(avg_t_action), avg_t_step[-1],
                np.mean(avg_t_step)))
            # do the learning
            for rk, agent in enumerate(self.agent_list):
                agent.update_buffer(prev_obs, agent.action, agent.curr_reward)

                if agent.num_total_steps % self.glb_update_freq == 0 or \
                    agent.done:
                    # update global and assign to local net
                    self._push_and_pull(rk, agent.done, agent.observation)

                if agent.done:  # done and print information
                    print('[glb ep {}][agent {}] done, ep reward [{}]'.format(
                        glb_num_episodes, rk, agent.episode_reward))
                    self.agent_reward_list[rk].append(agent.episode_reward)
                    self.glb_ep_reward_list.append(agent.episode_reward)
                    glb_num_episodes += 1

                agent.num_total_steps += 1
            # respawn dead agents
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list):
                if agent.done: respawn_rank_list.append(rk)
            if len(respawn_rank_list) > 0: # there're dead agents to respawn
                self.glb_env.reset(rank_list=respawn_rank_list)
                # update agent list
                for rk in respawn_rank_list:
                    self.agent_list[rk] = _PPO_Individual_Agent(
                        self.glb_env.ego_vehicle_list[rk],
                        glb_policy=self.glb_policy, rank=rk)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()


    def run(self):
        raise NotImplementedError('This agent does not use MP')

