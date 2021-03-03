import os
import time
import pickle
import copy
import torch
from threading import Thread, Lock
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import dist_utils as dist

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections import deque, defaultdict
from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG
from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.dashcam import (
    GlobalRecorder,
    TensorboardWriter,
    Visualizer,)


class SIG:
    GRAD_PUSH = 0
    PARAM_REQ = 1
    QUERY = 2
    GRAD = 3
    PARAM = 4


class _DPPO_Individual_Agent(Agent):
    def __init__(self, vehicle, local_policy, rank=None, memory=None, **kwargs):
        """A local individual Distributed PPO agent.
        Args:
            vehicle: ego-vehicle in env (e.g. env.vehicle_actor)
            local_policy: network shared by all PPO agents in a single env
            rank: an integer for identification of this local agent
            memory: PPO buffer
            **kwargs: include proximity_threshold=10.0,
                traffic_light_proximity_threshold=10.0,
                vehicle_proximity_threshold=15.0
        """
        super().__init__(vehicle, **kwargs)
        self.device = next(local_policy.parameters()).device
        self.local_policy = local_policy
        if memory is not None:
            self.memory = memory
        else:
            self.reset_memory()
        self.rank = rank
        self.done = False
        self.action = None
        self.rv_image = None
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        self.vehicle_actor = vehicle
        self.num_total_steps = 0
        self.episode_reward = 0
        self.curr_reward = 0
        self.observation = None
        self.termination_state = None

    def select_action(self, deterministic=False):
        prev_state = self.observation
        state_tensor = torch.from_numpy(prev_state).to(torch.float).to(self.device)
        action, logprob = self.local_policy.act(state_tensor,
            deterministic=deterministic)
        return action, logprob

    def update_local_policy(self):
        raise NotImplementedError

    def reset_memory(self):
        self.memory = {
            'action': [],
            'state': [],
            'logprob': [],
            'reward': [],
            'done': [],}


class DPPO_Server_Agent(object):
    def __init__(self, glb_policy, glb_optimizer, num_threads=1,
        glb_update_freq=1000, num_agents=1, max_glb_num_steps=1000000,
        gamma=.99, eps_clip=.2, optim_epochs=100, log_time='TEST',
        save_freq=100000, save_suffix='', verbose=False):
        """An asynchronous DPPO agent.
        Args:
            glb_policy: network shared by all PPO agents
            glb_optimizer: optimizer for the glb_policy
            max_glb_num_steps: max number of global steps
            gamma: reward discount factor
            eps_clip: clip parameter for PPO
            optim_epochs: update policy for how many epochs
            save_freq: checkpoint saving frequency
                (save the agent every N global steps)
            save_suffix: checkpoint saving suffix
            deterministic: evaluation mode (deterministic action)
            verbose: if print some debug information
        """
        self.glb_policy = glb_policy
        self.glb_optimizer = glb_optimizer
        self.num_agents = num_agents
        self.max_glb_num_steps = max_glb_num_steps
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.optim_epochs = optim_epochs
        # self.grad_clip = grad_clip
        self.device = next(glb_policy.parameters()).device
        self.model_len = len(parameters_to_vector(glb_policy.parameters()))
        self.glb_grad = torch.zeros(self.model_len, dtype=torch.float32, device=self.device, requires_grad=False)
        self.save_freq = save_freq
        ################################################################
        comm_vars = dist.init_param_server_comm()
        self.rank, self.world_size = comm_vars[:2]
        self.server_list, self.worker_list = comm_vars[2:4]
        self.server_group, self.worker_group = comm_vars[4:]
        self.num_servers = len(self.server_list)
        self.num_workers = len(self.worker_list)
        self.num_threads = num_threads
        self.vprint(comm_vars)
        # self.glb_num_steps = server_resources['glb_num_steps']
        # self.num_steps_since_update = server_resources['num_steps_since_update']
        # self.last_save_steps = server_resources['last_save_steps']
        # self.server_proc_lock = server_resources['server_proc_lock']
        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'DPPO{}x{}x{}{}'.format(self.num_servers,
            self.num_workers, self.num_agents, self.save_suffix)
        self.verbose = verbose
        self.recv_info_len = 4
        self.glb_ep_reward_list = []
        # self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.savetime = lambda: time.strftime('%b%d%I%M%p%S')
        self.glb_update_freq = glb_update_freq
        self.glb_num_steps = 1
        self.num_steps_since_update = 0
        self.last_save_steps = 0
        self.server_lock = Lock()
        self.server_save_lock = Lock()
        self.tb_log_dir = '{}/{}_{}'.format('./tensorboard_logs',
            self.run_name, log_time)
        self.tbwriter = None
        self.resumed = False

    def tb_write_config(self, tag, config):
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                filename_suffix='_{}'.format(self.run_name),)
        self.tbwriter.add_dict(tag, config)

    def vprint(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def to_tensor(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def listen(self):
        while self.glb_num_steps < self.max_glb_num_steps + 1:
            sender, num_steps_added, num_eps_added, signal = dist.recv(
                self.recv_info_len, tag=SIG.QUERY)
            self.vprint('server', self.rank, 'QUERY', sender, 
                num_steps_added, signal)
            if signal == SIG.GRAD_PUSH:
                _, vec_grad = dist.recv(self.recv_info_len, self.model_len, 
                    src=sender, tag=SIG.GRAD, device=self.device)
                # vector_to_parameters(vec_grad, self.glb_grad.parameters())
                with self.server_lock:
                    self.glb_grad += vec_grad
            elif signal == SIG.PARAM_REQ:
                self.vprint('server', self.rank, 'send param', sender, 
                    num_steps_added, signal)
                dist.isend(
                    [self.glb_num_steps, self.glb_num_episodes], 
                    self.glb_policy.parameters(), 
                    dst=sender, tag=SIG.PARAM
                ).wait()
            else:
                raise ValueError('signal not seen')
            with self.server_lock:
                self.glb_num_steps += num_steps_added
                self.num_steps_since_update += num_steps_added
                self.glb_num_episodes += num_eps_added
                if self.num_steps_since_update > self.glb_update_freq:
                    if self.resumed:
                        self.resumed = False
                    else:
                        print('[server rank {}][glb ep {}][glb step {}] updating ...'.format(
                            self.rank, self.glb_num_steps, self.glb_num_episodes,
                        ))
                        self.num_steps_since_update = 0
                        self.glb_optimizer.zero_grad()
                        ptr = 0
                        for param in self.glb_policy.parameters():
                            n = param.numel()
                            param._grad = self.glb_grad[ptr:ptr + n].view_as(param).data
                            ptr += n
                        self.glb_optimizer.step()
                        self.glb_grad = torch.zeros(self.model_len, 
                            dtype=torch.float32, device=self.device, 
                            requires_grad=False)

            # save checkpoint
            with self.server_save_lock:
                if self.glb_num_steps - self.last_save_steps > self.save_freq:
                    self.last_save_steps = self.glb_num_steps
                    self.save()

    def learn(self):
        thread_list = []
        for _ in range(self.num_threads):
            t = Thread(target=self.listen)
            thread_list.append(t)
        for t in thread_list: t.start()
        for t in thread_list: t.join()

    def test(self, videos=False):
        raise NotImplementedError

    def save(self, filename=None):
        if filename is None: filename = './ckpt{}_{}_{}.pth'.format(
            self.run_name, self.glb_num_steps, self.savetime())
        _ckpt = {
            'glb_policy': self.glb_policy.state_dict(),
            'glb_optimizer': self.glb_optimizer.state_dict(),
            'num_agents': self.num_agents,
            'max_glb_num_steps': self.max_glb_num_steps,
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'glb_update_freq': self.glb_update_freq,
            'optim_epochs': self.optim_epochs,
            'save_freq': self.save_freq,
            'verbose': self.verbose,
            'glb_num_steps': self.glb_num_steps,
            'num_steps_since_update': self.num_steps_since_update,
            'glb_num_episodes': self.glb_num_episodes,
        }
        torch.save(_ckpt, filename)
        print('checkpoint saved at [{}]'.format(filename))

    def load(self, checkpoint):
        self.glb_policy.load_state_dict(checkpoint['glb_policy'])
        self.glb_optimizer.load_state_dict(checkpoint['glb_optimizer'])
        print('checkpoint params loadeded')

    def resume(self, checkpoint, strict=False):
        if strict:
            assert self.num_agents == \
                checkpoint['num_agents'], '{} != {}'.format(
                self.num_agents, checkpoint['num_agents'])
        self.load(checkpoint)
        self.eps_clip = checkpoint['eps_clip']
        self.max_glb_num_steps = checkpoint['max_glb_num_steps']
        self.gamma = checkpoint['gamma']
        self.glb_update_freq = checkpoint['glb_update_freq']
        self.optim_epochs = checkpoint['optim_epochs']
        self.save_freq = checkpoint['save_freq']
        self.verbose = checkpoint['verbose']
        self.glb_num_steps = checkpoint['glb_num_steps']
        self.num_steps_since_update = checkpoint['num_steps_since_update']
        self.glb_num_episodes = checkpoint['glb_num_episodes']
        self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                purge_step=self.glb_num_episodes,
                filename_suffix='_{}'.format(self.run_name),)
        self.resumed = True

    def run(self):
        raise NotImplementedError('This agent does not use MP')


class DPPO_Worker_Agent(object):
    def __init__(self, local_env, local_policy, log_time='TEST',
        num_agents=1, max_glb_num_steps=1000000, gamma=.99, eps_clip=.2,
        grad_update_freq=1000, optim_epochs=100, focal_loss=False,
        grad_clip=None, save_freq=100000, save_suffix='', verbose=False):
        """An synchronous DPPO Worker agent.
        Args:
            local_env: the global environment
            local_policy: network shared by this local environment
            num_agents: number of PPO agents
            max_glb_num_steps: max number of global steps
            gamma: reward discount factor
            eps_clip: clip parameter for PPO
            grad_clip: value for clipping gradient, None to disable
            grad_update_freq: frequency of pushing gradients
            optim_epochs: update policy for how many epochs
            save_freq: checkpoint saving frequency
                (save the agent every N global steps)
            save_suffix: checkpoint saving suffix
            deterministic: evaluation mode (deterministic action)
            verbose: if print some debug information
        """
        self.local_env = local_env
        self.local_policy = local_policy
        self.model_len = len(parameters_to_vector(local_policy.parameters()))
        self.max_glb_num_steps = max_glb_num_steps
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.grad_update_freq = grad_update_freq
        self.optim_epochs = optim_epochs
        self.focal_loss = focal_loss
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.agent_reward_list = [[] for _ in self.rank_list]
        self.glb_ep_reward_list = []
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        ################################################################
        comm_vars = dist.init_param_server_comm()
        self.rank, self.world_size = comm_vars[:2]
        self.server_list, self.worker_list = comm_vars[2:4]
        self.server_group, self.worker_group = comm_vars[4:]
        self.num_servers = len(self.server_list)
        self.num_workers = len(self.worker_list)
        self.server_rank = (self.rank - self.num_servers) % self.num_servers
        self.vprint(comm_vars, self.server_rank)
        self.recv_info_len = 2
        self.grad_clip = grad_clip
        self.device = next(local_policy.parameters()).device
        self.save_freq = save_freq
        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'DPPO{}x{}x{}{}'.format(self.num_servers,
            self.num_workers, self.num_agents, self.save_suffix)
        self.verbose = verbose
        self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.savetime = lambda: time.strftime('%b%d%I%M%p%S')
        self.tb_log_dir = '{}/{}_{}'.format('./tensorboard_logs',
            self.run_name, log_time)
        self.glb_num_episodes = 1
        self.glb_num_steps = 0
        self.local_num_episodes = 1
        self.local_num_steps = 0
        self.num_steps_since_update = 0
        self.num_eps_since_update = 0
        self.recorder = GlobalRecorder
        self.tbwriter = None
        self.resumed = False

    def tb_write_config(self, tag, config):
        raise NotImplementedError

    def vprint(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def to_tensor(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def _update(self):
        rewards = []
        old_states = []
        old_actions = []
        old_logprobs = []
        for agent in self.agent_list:
            agent_rewards = deque()
            # Monte Carlo estimate of rewards:
            mem = agent.memory
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(mem['reward']), reversed(mem['done'])):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                agent_rewards.appendleft(discounted_reward)
            rewards.extend(list(agent_rewards))
            old_states.extend(mem['state'])
            old_actions.extend(mem['action'])
            old_logprobs.extend(mem['logprob'])

        # Normalizing the rewards:
        # rewards = torch.tensor(rewards).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # print('rewards', rewards, rewards.shape)

        # convert list to tensor
        # print(old_states)
        old_states = torch.tensor(old_states, dtype=torch.float32,
            device=self.device).squeeze().detach()
        # print(old_states.shape)
        old_actions = torch.tensor(old_actions, dtype=torch.float32,
            device=self.device).squeeze().detach()
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32,
            device=self.device).squeeze().detach()

        # Optimize policy for K epochs:
        total_loss = None
        for _ in range(self.optim_epochs):
            # Evaluating old actions and values:
            # print(old_states.shape)
            logprobs, state_values, dist_entropy = self.local_policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            # print('state_values', state_values.shape)
            advantages = rewards - state_values.detach()
            if self.focal_loss:
                _al, _ga = self.focal_loss # assume a [alpha, gamma] list
                _p = torch.exp(logprobs)
                _focal_loss = -_al * ((1 - _p) ** (_ga - 1)) * \
                    (_p * _ga * logprobs + _p - 1)
                advantages = advantages * _focal_loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values,
                rewards) - 0.01 * dist_entropy

            # send gradient
            if total_loss is None:
                total_loss = loss.mean() / self.optim_epochs
            else:
                total_loss += loss.mean() / self.optim_epochs

        total_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.local_policy.parameters(), self.grad_clip)

        # send gradients
        self.send_gradients()
        # get new parameters
        self.glb_num_steps, self.glb_num_episodes = self.update_parameters()

        # zero grad
        for p in self.local_policy.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

        for agent in self.agent_list:
            if agent.done: continue # no need to update for a done agent
            agent.reset_memory()

    def send_gradients(self):
        self.vprint('rank', self.rank, 'send_gradients')
        param_grad = [item.grad for item in self.local_policy.parameters()]
        vec_grad = parameters_to_vector(param_grad).detach()
        overhead = [self.rank, self.num_steps_since_update, 
            self.num_eps_since_update, SIG.GRAD_PUSH]
        dist.isend(overhead, dst=self.server_rank, tag=SIG.QUERY).wait()
        dist.isend(overhead, vec_grad, dst=self.server_rank, tag=SIG.GRAD).wait()

    def update_parameters(self):
        self.vprint('rank', self.rank, 'update_parameters')
        overhead = [self.rank, self.num_steps_since_update, 
            self.num_eps_since_update, SIG.PARAM_REQ]
        dist.isend(overhead, dst=self.server_rank, tag=SIG.QUERY).wait()
        glb_steps, glb_eps, vec_param = dist.recv(self.recv_info_len, self.model_len,
            src=self.server_rank, tag=SIG.PARAM, device=self.device)
        vector_to_parameters(vec_param, self.local_policy.parameters())
        return glb_steps, glb_eps

    def learn(self):
        # get new parameters
        self.glb_num_steps, self.glb_num_episodes = self.update_parameters()
        # init tensorboard
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                filename_suffix='_{}'.format(self.run_name),)
        # initialize
        self.local_env.reset(rank_list=self.rank_list)
        self.local_env.spawn_npc_vehicles(51 - self.num_agents)
        self.agent_list = [_DPPO_Individual_Agent(
            self.local_env.ego_vehicle_list[i],
            local_policy=self.local_policy, rank=i) for i in self.rank_list]
        self.local_env.reset_vehicle_agent(self.agent_list)
        self.local_env.step()

        avg_t_action, avg_t_step  = [], []

        while self.glb_num_steps < self.max_glb_num_steps + 1:
            # take action
            ts_action = time.time()
            for rk, agent in enumerate(self.agent_list):
                # prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                action, logprob = agent.select_action()
                agent.action = action
                # update partial memory
                agent.memory['state'].append(agent.observation.tolist())
                agent.memory['action'].append(action.tolist())
                agent.memory['logprob'].append(logprob.tolist())
            te_action = time.time()
            self.vprint('action chosen:', [a.action for a in self.agent_list])
            # get new observation
            ts_step = time.time()
            self.local_env.step()
            te_step = time.time()
            avg_t_action.append(te_action - ts_action)
            avg_t_step.append(te_step - ts_step)
            self.vprint('[num_agent {}][action time {:.4f}, avg {:.4f}]'
                '[step time {:.4f}, avg {:.4f}]'.format(self.num_agents,
                avg_t_action[-1], np.mean(avg_t_action), avg_t_step[-1],
                np.mean(avg_t_step)))

            for rk, agent in enumerate(self.agent_list):
                agent.memory['reward'].append(agent.curr_reward)
                agent.memory['done'].append(agent.done)

                if agent.done:  # done and print information
                    print('[{}]'.format(self.time()) + \
                        '[{}][rank {}]'.format(self.run_name, self.rank) + \
                        '[local ep {}][local step {}][agent {}] done({})'
                        ', ep reward [{:.4f}]'.format(
                        self.local_num_episodes, self.local_num_steps, rk,
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
                    self.recorder['recent']['avg_dist_to_trgt'].record_value(
                        agent.episode_measurements['distance_to_goal_trajec'])
                    self.recorder['recent']['success_rate'].record_value(
                        success_int)
                    self.recorder['recent']['collision_rate'].record_value(
                        obs_collision_int)
                    # tensorboard
                    self.tbwriter.add_scalar('rank_{}/episode/reward'.format(self.rank),
                        agent.episode_reward, self.glb_num_episodes)
                    self.tbwriter.add_scalar('episode/dist_to_target',
                        agent.episode_measurements['distance_to_goal_trajec'],
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/episode/num_collisions'.format(self.rank),
                        agent.episode_measurements['num_collisions'],
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/episode/success_rate'.format(self.rank),
                        self.recorder['train']['success_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/episode/collision_rate'.format(self.rank),
                        self.recorder['train']['collision_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/episode/avg_reward'.format(self.rank),
                        self.recorder['train']['avg_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/episode/max_reward'.format(self.rank),
                        self.recorder['train']['max_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/episode/dist_to_target'.format(self.rank),
                        self.recorder['episode']['dist_to_target'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/recent/avg_reward'.format(self.rank),
                        self.recorder['recent']['avg_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/recent/max_reward'.format(self.rank),
                        self.recorder['recent']['max_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rrank_{}/ecent/min_reward'.format(self.rank),
                        self.recorder['recent']['min_reward'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/recent/avg_dist_to_target'.format(self.rank),
                        self.recorder['recent']['avg_dist_to_trgt'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/recent/success_rate'.format(self.rank),
                        self.recorder['recent']['success_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('rank_{}/recent/collision_rate'.format(self.rank),
                        self.recorder['recent']['collision_rate'].summary(),
                        self.glb_num_episodes)
                    self.recorder.summary_all()
                    self.local_num_episodes += 1
                    self.num_eps_since_update += 1

                agent.num_total_steps += 1
                self.num_steps_since_update += 1
                self.local_num_steps += 1

            if self.num_steps_since_update >= self.grad_update_freq:
                if self.resumed:
                    # skip the first update after resume
                    self.resumed = False
                else:
                    self._update()
                self.num_steps_since_update = 0
                self.num_eps_since_update = 0

            # respawn dead agents
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list):
                if agent.done: respawn_rank_list.append(rk)
            if len(respawn_rank_list) > 0: # there're dead agents to respawn
                self.local_env.reset(rank_list=respawn_rank_list)
                # update agent list
                for rk in respawn_rank_list:
                    self.agent_list[rk] = _DPPO_Individual_Agent(
                        self.local_env.ego_vehicle_list[rk],
                        local_policy=self.local_policy, rank=rk,
                        memory=self.agent_list[rk].memory)
                self.local_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                # self.local_env.step()

    def test(self, videos=False):
        raise NotImplementedError

    def save(self, filename=None):
        raise NotImplementedError

    def load(self, checkpoint):
        self.local_policy.load_state_dict(checkpoint['local_policy'])
        # self.glb_optimizer.load_state_dict(checkpoint['glb_optimizer'])
        print('checkpoint params loadeded')

    def resume(self, checkpoint, strict=False):
        if strict:
            assert self.num_agents == \
                checkpoint['num_agents'], '{} != {}'.format(
                self.num_agents, checkpoint['num_agents'])
        self.load(checkpoint)
        self.eps_clip = checkpoint['eps_clip']
        self.max_glb_num_steps = checkpoint['max_glb_num_steps']
        self.gamma = checkpoint['gamma']
        # self.glb_update_freq = checkpoint['glb_update_freq']
        self.optim_epochs = checkpoint['optim_epochs']
        self.save_freq = checkpoint['save_freq']
        self.verbose = checkpoint['verbose']
        self.glb_num_steps = checkpoint['glb_num_steps']
        # self.num_steps_since_update = checkpoint['num_steps_since_update']
        self.glb_num_episodes = checkpoint['glb_num_episodes']
        self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                purge_step=self.glb_num_episodes,
                filename_suffix='_{}'.format(self.run_name),)
        self.resumed = True

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

    glb_policy = PPOActorCritic_Continuous(N_S, N_A) # global network

    print('*' * 80)
    print('FINISHED')
    print('*' * 80)
