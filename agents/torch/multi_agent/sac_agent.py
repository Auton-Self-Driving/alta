import os
import random
import time
import copy
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from collections import deque
from torch.distributions import Normal
from network import SoftQNetwork, PolicyNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG
from environment.carla_9_4.agents.navigation.agent import Agent


class VanillaReplayBuffer(object):
    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        for exp in batch:
            state, action, reward, next_state, done = exp
            state_batch.append(torch.tensor(state, dtype=torch.float).squeeze())
            action_batch.append(torch.tensor(action, dtype=torch.float).squeeze())
            reward_batch.append(torch.tensor(reward, dtype=torch.float).squeeze())
            next_state_batch.append(torch.tensor(next_state, dtype=torch.float).squeeze())
            done_batch.append(torch.tensor(done, dtype=torch.float).squeeze())

        return (state_batch, action_batch, \
            reward_batch, next_state_batch, done_batch)


class _SAC_Individual_Agent(Agent):
    def __init__(self, vehicle, glb_policy, rank=None, **kwargs):
        """A local individual SAC agent.
        Args:
            vehicle: ego-vehicle in env (e.g. env.vehicle_actor)
            glb_policy: policy network for selecting action
            rank: an integer for identification of this local agent
            **kwargs: include proximity_threshold=10.0,
                traffic_light_proximity_threshold=10.0,
                vehicle_proximity_threshold=15.0
        """
        super().__init__(vehicle, **kwargs)
        self.device = next(glb_policy.parameters()).device
        self.glb_policy = glb_policy
        self.rank = rank
        self.done = False
        self.action = None
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        self.vehicle_actor = vehicle
        self.num_total_steps = 0
        self.episode_reward = 0
        self.curr_reward = 0
        self.observation = None
        self.termination_state = None

    def select_action(self):
        prev_state = self.observation
        state_tensor = torch.from_numpy(prev_state).to(torch.float).to(self.device)
        mean, log_std = self.glb_policy(state_tensor)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        # return self.glb_policy.rescale_action(action)
        return action


class SAC_Collective_Agent(object):
    def __init__(self, glb_env, glb_q1, q1_optimizer, glb_q2, q2_optimizer, 
        glb_policy, policy_optimizer, log_alpha, alpha_optimizer,
        target_entropy, buffer, num_agents=1, tau=0.01, batch_size=512, 
        max_glb_num_episodes=10000, gamma=.99, q_update_freq=25,
        target_update_freq=2, verbose=False):
        """An synchronous SAC agent.
        Args:
            glb_env: the global environment
            glb_q1: the first global q_net
            q1_optimizer: optimizer for glb_q1
            glb_q2: the second global q_net
            q2_optimizer: optimizer for glb_q2
            glb_policy: local policy net for individual SAC agents
            policy_optimizer: optimizer for glb_policy
            log_alpha:
            alpha_optimizer:
            target_entropy:
            buffer: replay buffer
            num_agents: number of SAC agents
            tau:
            batch_size:
            max_glb_num_episodes: max number of global episodes
            gamma: reward discount factor
            q_update_freq: update frequency of q networks 
                (update q networks every N steps)
            target_update_freq: update frequency of target q networks 
                (update target and policy every N q-updates)
            optim_epochs: update policy for how many epochs
            verbose: if print some debug information
        """
        super().__init__()
        self.glb_env = glb_env
        self.glb_q1 = glb_q1
        self.q1_optimizer = q1_optimizer
        self.target_q1 = copy.deepcopy(self.glb_q1)
        self.glb_q2 = glb_q2
        self.q2_optimizer = q2_optimizer
        self.target_q2 = copy.deepcopy(self.glb_q2)
        self.glb_policy = glb_policy
        self.policy_optimizer = policy_optimizer
        self.buffer = buffer
        self.log_alpha = log_alpha
        self.alpha_optimizer = alpha_optimizer
        self.target_entropy = target_entropy
        self.batch_size = batch_size
        self.max_glb_num_episodes = max_glb_num_episodes
        self.gamma = gamma
        self.tau = tau
        self.q_update_freq = q_update_freq
        self.target_update_freq = target_update_freq
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        self.device = next(glb_policy.parameters()).device
        self.verbose = verbose
        self.glb_ep_reward_list = []
        self.agent_reward_list = [[] for _ in self.rank_list]
        self.num_q_updates_since_target_update = 0

    def vprint(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def to_tensor(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def _update(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).view(-1, 1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).view(-1, 1).to(self.device)
        # print('171', states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        next_actions, next_log_pi = self.glb_policy.sample(next_states)
        # print('173', next_actions.shape, next_log_pi.shape)
        next_q1 = self.target_q1(next_states, next_actions)
        next_q2 = self.target_q2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.log_alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.glb_q1.forward(states, actions)
        curr_q2 = self.glb_q2.forward(states, actions)        
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        self.num_q_updates_since_target_update += 1

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.glb_policy.sample(states)

        if self.num_q_updates_since_target_update % self.target_update_freq == 0:
            min_q = torch.min(
                self.glb_q1.forward(states, new_actions),
                self.glb_q2.forward(states, new_actions)
            )
            policy_loss = (self.log_alpha.exp() * log_pi - min_q).mean()
            # update policy networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            # update target networks
            self.update_target_q()
            
        # update alpha temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def learn(self):
        glb_num_episodes = 1
        num_steps_since_update = glb_num_steps = 0
        # initialize
        self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.spawn_npc_vehicles()
        self.agent_list = [_SAC_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()

        avg_t_action, avg_t_step  = [], []

        while glb_num_episodes < self.max_glb_num_episodes + 1:
            # take action
            ts_action = time.time()
            for rk, agent in enumerate(self.agent_list):
                action = agent.select_action()
                agent.prev_state = agent.observation
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

            for rk, agent in enumerate(self.agent_list):
                # push into the buffer
                self.buffer.append(agent.prev_state, agent.action, agent.curr_reward,
                    agent.observation, int(agent.done))

                if agent.done:  # done and print information
                    print('[glb ep {}][glb step {}][agent {}] done({})'
                        ', ep reward [{}]'.format(
                        glb_num_episodes, glb_num_steps, rk, 
                        agent.termination_state, agent.episode_reward))
                    self.agent_reward_list[rk].append(agent.episode_reward)
                    self.glb_ep_reward_list.append(agent.episode_reward)
                    glb_num_episodes += 1

                agent.num_total_steps += 1
                num_steps_since_update += 1
                glb_num_steps += 1

                if num_steps_since_update >= self.q_update_freq and \
                    len(self.buffer) > self.batch_size:
                    # do the learning
                    # print('updating policy...')
                    self._update()
                    num_steps_since_update = 0

            # respawn dead agents
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list):
                if agent.done: respawn_rank_list.append(rk)
            if len(respawn_rank_list) > 0: # there're dead agents to respawn
                self.glb_env.reset(rank_list=respawn_rank_list)
                # update agent list
                for rk in respawn_rank_list:
                    self.agent_list[rk] = _SAC_Individual_Agent(
                        self.glb_env.ego_vehicle_list[rk],
                        glb_policy=self.glb_policy, rank=rk)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()

    def update_target_q(self):
        for target_param, param in zip(self.target_q1.parameters(), self.glb_q1.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_q2.parameters(), self.glb_q2.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def run(self):
        raise NotImplementedError('This agent does not use MP')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["OMP_NUM_THREADS"] = '1'

    env = CarlaEnv(ENV_CONFIG)

    N_S = env.observation_space.shape[-1]
    N_A = env.action_space.shape[-1]
    print(N_S, N_A)
    # from IPython import embed; embed()

    glb_policy = PolicyNetwork(N_S, N_A) # global network

    print('*' * 80)
    print('FINISHED')
    print('*' * 80)
