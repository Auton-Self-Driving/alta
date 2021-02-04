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
from environment.carla_9_4.dashcam import GlobalRecorder, TensorboardWriter


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
        self.step_reward = 0
        self.observation = None
        self.termination_state = None

    def select_action(self, deterministic=False):
        prev_state = self.observation
        state_tensor = torch.from_numpy(prev_state).to(torch.float).to(self.device)
        mean, log_std = self.glb_policy(state_tensor)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.mean if deterministic else normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        # return self.glb_policy.rescale_action(action)
        return action


class SAC_Collective_Agent(object):
    def __init__(self, glb_env, glb_q1, q1_optimizer, glb_q2, q2_optimizer,
        glb_policy, policy_optimizer, log_alpha, alpha_optimizer,
        target_entropy, buffer, num_agents=1, tau=0.01, batch_size=512,
        max_glb_num_steps=1000000, gamma=.99, q_update_freq=1,
        target_update_freq=1, save_freq=100000, save_suffix='', verbose=False):
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
            max_glb_num_steps: max number of global steps
            gamma: reward discount factor
            q_update_freq: update frequency of q networks
                (update q networks every N steps)
            target_update_freq: update frequency of target q networks
                (update target and policy every N q-updates)
            save_freq: checkpoint saving frequency
                (save the agent every N global steps)
            save_suffix: checkpoint saving suffix
            deterministic: evaluation mode (deterministic action)
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
        self.max_glb_num_steps = max_glb_num_steps
        self.gamma = gamma
        self.tau = tau
        self.q_update_freq = q_update_freq
        self.target_update_freq = target_update_freq
        self.save_freq = save_freq
        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'SACx{}{}'.format(self.num_agents,
            self.save_suffix)
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        self.device = next(glb_policy.parameters()).device
        self.verbose = verbose
        self.glb_ep_reward_list = []
        self.agent_reward_list = [[] for _ in self.rank_list]
        self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.num_q_upd_since_target_upd = 0
        self.glb_num_episodes = 1
        self.num_steps_since_update = 0
        self.glb_num_steps = 0
        self.recorder = GlobalRecorder
        self.tbwriter = None

    def tb_write_config(self, tag, config):
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir='./tensorboard_logs/',
                filename_suffix='_{}'.format(self.run_name),)
        self.tbwriter.add_dict(tag, config)

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

        self.num_q_upd_since_target_upd += 1

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.glb_policy.sample(states)

        if self.num_q_upd_since_target_upd % self.target_update_freq == 0:
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
        alpha_loss = (self.log_alpha * (-log_pi - \
            self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # print('self.log_alpha:', self.log_alpha)

    def learn(self):
        # initialize
        # initialize
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir='./tensorboard_logs/',
                filename_suffix='_{}'.format(self.run_name),)
        self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.spawn_npc_vehicles()
        self.agent_list = [_SAC_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()

        avg_t_action, avg_t_step  = [], []

        while self.glb_num_steps < self.max_glb_num_steps + 1:
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
                self.buffer.append(agent.prev_state, agent.action,
                    agent.step_reward, agent.observation, int(agent.done))

                if agent.done:  # done and print information
                    print('[{}]'.format(self.time()) + \
                        '[{}]'.format(self.run_name) + \
                        '[glb ep {}][glb step {}][agent {}] done({})'
                        ', ep reward [{:.4f}]'.format(
                        self.glb_num_episodes, self.glb_num_steps, rk,
                        agent.termination_state, agent.episode_reward))
                    self.agent_reward_list[rk].append(agent.episode_reward)
                    self.glb_ep_reward_list.append(agent.episode_reward)
                    success_int = int('success' == agent.termination_state)
                    obs_collision_int = int('obs_collision' == agent.termination_state)
                    # record statistics
                    self.recorder['train']['reward'].record_value(
                        agent.episode_reward)
                    self.recorder['train']['max_reward'].record_value(
                        agent.episode_reward)
                    self.recorder['train']['avg_reward'].record_value(
                        agent.episode_reward)
                    self.recorder['train']['dist_to_target'].record_value(
                        agent.episode_measurements['distance_to_goal_trajec'])
                    self.recorder['train']['num_collisions'].record_value(
                        agent.episode_measurements['num_collisions'])
                    self.recorder['train']['success_rate'].record_value(
                        success_int)
                    self.recorder['train']['collision_rate'].record_value(
                        obs_collision_int)
                    self.recorder['episode']['dist_to_target'].record_value(
                        agent.episode_measurements['distance_to_goal_trajec'])
                    self.recorder['recent']['avg_reward'].record_value(
                        agent.episode_reward)
                    self.recorder['recent']['max_reward'].record_value(
                        agent.episode_reward)
                    self.recorder['recent']['min_reward'].record_value(
                        agent.episode_reward)
                    self.recorder['recent']['mean_dist_to_trgt'].record_value(
                        agent.episode_measurements['distance_to_goal_trajec'])
                    self.recorder['recent']['success_rate'].record_value(
                        success_int)
                    self.recorder['recent']['collision_rate'].record_value(
                        obs_collision_int)
                    # tensorboard_recording
                    self.tbwriter.add_scalar('episode/reward',
                        agent.episode_reward, self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/dist_to_target',
                        agent.episode_measurements['distance_to_goal_trajec'],
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/num_collisions',
                        agent.episode_measurements['num_collisions'],
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/success_rate',
                        self.recorder['train']['success_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/collision_rate',
                        self.recorder['train']['collision_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/avg_reward',
                        self.recorder['train']['avg_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/max_reward',
                        self.recorder['train']['max_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/dist_to_target',
                        self.recorder['episode']['dist_to_target'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/avg_reward',
                        self.recorder['recent']['avg_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/max_reward',
                        self.recorder['recent']['max_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/min_reward',
                        self.recorder['recent']['min_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/mean_dist_to_target',
                        self.recorder['recent']['mean_dist_to_trgt'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/success_rate',
                        self.recorder['recent']['success_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/collision_rate',
                        self.recorder['recent']['collision_rate'].summary(),
                        self.glb_num_episodes)
                    self.recorder.summary_all()
                    self.glb_num_episodes += 1


                agent.num_total_steps += 1
                self.num_steps_since_update += 1
                self.glb_num_steps += 1
                # save checkpoint
                if self.glb_num_steps % self.save_freq == 0:
                    self.save()

                if self.num_steps_since_update >= self.q_update_freq and \
                    len(self.buffer) > self.batch_size:
                    # do the learning
                    # print('updating policy...')
                    self._update()
                    self.num_steps_since_update = 0

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

    def test(self):
        # initialize
        idx_list = list(range(self.num_agents))
        self.glb_num_test_episodes = self.glb_env.config['num_episodes']
        self.glb_env.reset(rank_list=self.rank_list, use_idx=True,
            idx_list=idx_list, reset_npc=True)
        self.glb_env.spawn_npc_vehicles()
        self.agent_list = [_SAC_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()

        self.num_successes = 0
        while self.glb_num_episodes < self.glb_num_test_episodes + 1:
            # take action
            for rk, agent in enumerate(self.agent_list):
                # prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                action = agent.select_action(deterministic=True)
                agent.action = action
            self.vprint('action chosen:', [a.action for a in self.agent_list])
            # get new observation
            self.glb_env.step()

            for rk, agent in enumerate(self.agent_list):
                if agent.done:  # done and print information
                    if agent.termination_state == 'success':
                        self.num_successes += 1
                    print('[test {}][glb ep {}/{}]'.format(
                        self.run_name,
                        self.glb_num_episodes,
                        self.glb_num_test_episodes) + \
                        '[score {:.2%}][glb step {}][agent {}] done({})'
                        ', ep reward [{:.4f}]'.format(
                        self.num_successes / self.glb_num_test_episodes,
                        self.glb_num_steps, rk,
                        agent.termination_state, agent.episode_reward))
                    self.agent_reward_list[rk].append(agent.episode_reward)
                    self.glb_ep_reward_list.append(agent.episode_reward)
                    self.glb_num_episodes += 1

                agent.num_total_steps += 1
                self.glb_num_steps += 1

            # respawn dead agents
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list):
                if agent.done and self.num_agents + \
                    idx_list[rk] < self.glb_num_test_episodes:
                    respawn_rank_list.append(rk)
                    idx_list[rk] += self.num_agents
            if len(respawn_rank_list) > 0: # there're dead agents to respawn
                self.glb_env.reset(rank_list=respawn_rank_list, use_idx=True,
                    idx_list=idx_list, reset_npc=True)
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

    def save(self, filename=None):
        if filename is None: filename = './ckpt{}_{}_{}.pth'.format(
            self.run_name, self.glb_num_steps, time.strftime('%b%d%I%M%p%S'))
        _ckpt = {
            'glb_q1': self.glb_q1.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'glb_q2': self.glb_q2.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'glb_policy': self.glb_policy.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'target_entropy': self.target_entropy,
            'num_agents': self.num_agents,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'max_glb_num_steps': self.max_glb_num_steps,
            'gamma': self.gamma,
            'q_update_freq': self.q_update_freq,
            'target_update_freq': self.target_update_freq,
            'save_freq': self.save_freq,
            'verbose': self.verbose,
            'glb_num_steps': self.glb_num_steps,
            'num_steps_since_update': self.num_steps_since_update,
            'glb_num_episodes': self.glb_num_episodes,
            'num_q_upd_since_target_upd': self.num_q_upd_since_target_upd,
        }
        torch.save(_ckpt, filename)
        print('checkpoint saved at [{}]'.format(filename))

    def load(self, checkpoint):
        self.glb_q1.load_state_dict(checkpoint['glb_q1'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.glb_q2.load_state_dict(checkpoint['glb_q2'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.glb_policy.load_state_dict(checkpoint['glb_policy'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        print('checkpoint params loadeded')

    def resume(self, checkpoint, strict=False):
        if strict:
            assert self.num_agents == \
                checkpoint['num_agents'], '{} != {}'.format(
                self.num_agents, checkpoint['num_agents'])
        self.load(checkpoint)
        self.target_entropy = checkpoint['target_entropy']
        self.tau = checkpoint['tau']
        self.batch_size = checkpoint['batch_size']
        self.max_glb_num_steps = checkpoint['max_glb_num_steps']
        self.gamma = checkpoint['gamma']
        self.q_update_freq = checkpoint['q_update_freq']
        self.target_update_freq = checkpoint['target_update_freq']
        self.save_freq = checkpoint['save_freq']
        self.verbose = checkpoint['verbose']
        self.glb_num_steps = checkpoint['glb_num_steps']
        self.num_steps_since_update = checkpoint['num_steps_since_update']
        self.glb_num_episodes = checkpoint['glb_num_episodes']
        self.num_q_upd_since_target_upd = \
            checkpoint['num_q_upd_since_target_upd']
        self.tbwriter = TensorboardWriter(
            log_dir='./tensorboard_logs/',
            purge_step=self.glb_num_episodes,
                filename_suffix='_{}'.format(self.run_name),)

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
