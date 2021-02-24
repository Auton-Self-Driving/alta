import os
import time
import pickle
import copy
import torch
import torch.multiprocessing as mp
from threading import Thread, Lock, Condition, Barrier
import torch.nn.functional as F
import numpy as np

from collections import deque, defaultdict
from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG
from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.dashcam import (
    GlobalRecorder,
    TensorboardWriter,
    Visualizer,)

class _PPO_Individual_Agent(Agent):
    def __init__(self, vehicle, glb_policy, rank=None, memory=None, **kwargs):
        """A local individual PPO agent.
        Args:
            vehicle: ego-vehicle in env (e.g. env.vehicle_actor)
            glb_policy: network shared by all PPO agents
            rank: an integer for identification of this local agent
            memory: PPO buffer
            **kwargs: include proximity_threshold=10.0,
                traffic_light_proximity_threshold=10.0,
                vehicle_proximity_threshold=15.0
        """
        super().__init__(vehicle, **kwargs)
        self.device = next(glb_policy.parameters()).device
        self.glb_policy = glb_policy
        self.local_policy = pickle.loads(pickle.dumps(self.glb_policy))
        self.local_policy = self.local_policy.to(self.device)
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
        self.local_policy.load_state_dict(self.glb_policy.state_dict())

    def reset_memory(self):
        self.memory = {
            'action': [],
            'state': [],
            'logprob': [],
            'reward': [],
            'done': [],}


class PPO_Collective_Agent(object):
    def __init__(self, glb_env, glb_policy, glb_optimizer, num_agents=1,
        max_glb_num_steps=1000000, gamma=.99, eps_clip=.2, nesterov=False,
        glb_update_freq=1000, optim_epochs=100, focal_loss=False,
        grad_clip=None, save_freq=100000, save_suffix='', verbose=False):
        """An synchronous PPO agent.
        Args:
            glb_env: the global environment
            glb_policy: network shared by all PPO agents
            glb_optimizer: optimizer for the glb_policy
            num_agents: number of PPO agents
            max_glb_num_steps: max number of global steps
            gamma: reward discount factor
            eps_clip: clip parameter for PPO
            grad_clip: value for clipping gradient, None to disable
            nesterov: if using nesterov update
            glb_update_freq: update frequency of glb_policy
            optim_epochs: update policy for how many epochs
            save_freq: checkpoint saving frequency
                (save the agent every N global steps)
            save_suffix: checkpoint saving suffix
            deterministic: evaluation mode (deterministic action)
            verbose: if print some debug information
        """
        super().__init__()
        self.glb_env = glb_env
        self.glb_policy = glb_policy
        self.glb_optimizer = glb_optimizer
        self.max_glb_num_steps = max_glb_num_steps
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.glb_update_freq = glb_update_freq
        self.optim_epochs = optim_epochs
        self.focal_loss = focal_loss
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        self.grad_clip = grad_clip
        self.nesterov = nesterov
        self.device = next(glb_policy.parameters()).device
        self.save_freq = save_freq
        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'PPOx{}{}'.format(self.num_agents,
            self.save_suffix)
        self.verbose = verbose
        self.glb_ep_reward_list = []
        self.agent_reward_list = [[] for _ in self.rank_list]
        self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.savetime = lambda: time.strftime('%b%d%I%M%p%S')
        self.tb_log_dir = '{}/{}_{}'.format('./tensorboard_logs',
            self.run_name, self.savetime())
        self.vid_log_dir = '{}/{}_{}'.format('./video_logs',
            'PPO_test', self.savetime())
        self.glb_num_episodes = 1
        self.glb_num_steps = 0
        self.num_steps_since_update = 0
        self.recorder = GlobalRecorder
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
        for _ in range(self.optim_epochs):
            # Evaluating old actions and values:
            # print(old_states.shape)
            logprobs, state_values, dist_entropy = self.glb_policy.evaluate(
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

            # take gradient step
            self.glb_optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.glb_policy.parameters(), self.grad_clip)
            self.glb_optimizer.step()

        for agent in self.agent_list:
            if agent.done: continue # no need to update for a done agent
            agent.update_local_policy()
            agent.reset_memory()

    def _nesterov_update(self):
        # _glb_policy_state = copy.deepcopy(self.glb_policy.state_dict())
        # _glb_optim_state = copy.deepcopy(self.glb_optimizer.state_dict())
        # _local_policy = copy.deepcopy(self.glb_policy)
        # _local_optim = copy.deepcopy(self.glb_optimizer)

        rewards, old_states, old_actions, old_logprobs = [], [], [], []
        list_rewards, list_old_states, list_old_actions = [], [], []
        list_old_logprobs, list_local_optim, rank_mask = [], [], []
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
                rank_mask.append(agent.rank)
            rewards.extend(list(agent_rewards))
            old_states.extend(mem['state'])
            old_actions.extend(mem['action'])
            old_logprobs.extend(mem['logprob'])

            _agt_reward = torch.tensor(list(agent_rewards),
                dtype=torch.float32, device=self.device)
            _agt_reward = (_agt_reward - _agt_reward.mean()) / \
                (_agt_reward.std() + 1e-5)
            _agt_states = torch.tensor(mem['state'], dtype=torch.float32,
                device=self.device).squeeze().detach()
            _agt_actions = torch.tensor(mem['action'] , dtype=torch.float32,
                device=self.device).squeeze().detach()
            _agt_logprobs = torch.tensor(mem['logprob'] , dtype=torch.float32,
                device=self.device).squeeze().detach()
            list_rewards.append(_agt_reward)
            list_old_states.append(_agt_states)
            list_old_actions.append(_agt_actions)
            list_old_logprobs.append(_agt_logprobs)
            _local_optim = self.glb_optimizer.__class__(
                agent.local_policy.parameters(), lr=999.9)
            _local_optim.load_state_dict(self.glb_optimizer.state_dict())
            list_local_optim.append(_local_optim)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.tensor(old_states, dtype=torch.float32,
            device=self.device).squeeze().detach()
        old_actions = torch.tensor(old_actions, dtype=torch.float32,
            device=self.device).squeeze().detach()
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32,
            device=self.device).squeeze().detach()
        rank_mask = torch.tensor(rank_mask, dtype=torch.uint8,
            device=self.device).squeeze().detach()

        for _ in range(self.optim_epochs):
            # glb_ratios = torch.zeros_like(rewards, dtype=torch.float32,
            #     device=self.device, requires_grad=False)
            glb_advantages = torch.zeros_like(rewards, dtype=torch.float32,
                device=self.device, requires_grad=False)

            self.glb_optimizer.zero_grad()

            for (agent, _agt_reward, _agt_states,
                _agt_actions, _agt_logprobs, _local_optim) in zip(
                self.agent_list, list_rewards, list_old_states,
                list_old_actions, list_old_logprobs, list_local_optim,
            ):
                # independent updates
                if len(_agt_reward) == 0: continue

                _logprobs, _state_values, _dist_entropy = \
                    agent.local_policy.evaluate(_agt_states, _agt_actions)

                _ratios = torch.exp(_logprobs - _agt_logprobs.detach())
                _advantages = _agt_reward - _state_values.detach()
                # add to glb_ratios & glb_advantages
                # glb_ratios[rank_mask == agent.rank] += _ratios
                glb_advantages[rank_mask == agent.rank] += _advantages
                _surr1 = _ratios * _advantages
                _surr2 = torch.clamp(_ratios, 1 - self.eps_clip,
                    1 + self.eps_clip) * _advantages
                _loss = -torch.min(_surr1, _surr2) + 0.5 * F.mse_loss(
                    _state_values, _agt_reward) - 0.01 * _dist_entropy

                # take gradient step
                _local_optim.zero_grad()
                _loss = _loss.mean()
                _loss.backward()
                # accumulate loss to glb_policy
                for lp, gp in zip(
                    agent.local_policy.parameters(),
                    self.glb_policy.parameters(),
                ):
                    if gp._grad is None:
                        gp._grad = lp.grad.clone() / self.num_agents
                    else:
                        gp._grad += lp.grad / self.num_agents


                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        agent.local_policy.parameters(), self.grad_clip,)
                _local_optim.step()

                # calculate global surr1 and surr2
                if len(rank_mask[rank_mask != agent.rank]) > 0:
                    # _logprobs, _state_values, _dist_entropy = \
                    # with torch.no_grad:
                    _logprobs, _state_values, _dist_entropy = \
                        agent.local_policy.evaluate(
                        old_states[rank_mask != agent.rank],
                        old_actions[rank_mask != agent.rank],
                    )
                    _ratios = torch.exp(_logprobs - \
                        old_logprobs[rank_mask != agent.rank].detach())
                    _advantages = rewards[rank_mask != agent.rank] - \
                        _state_values.detach()
                    _surr1 = _ratios * _advantages
                    _surr2 = torch.clamp(_ratios, 1 - self.eps_clip,
                        1 + self.eps_clip) * _advantages
                    _loss = -torch.min(_surr1, _surr2) + 0.5 * F.mse_loss(
                        _state_values, rewards[rank_mask != agent.rank]) - \
                        0.01 * _dist_entropy
                    _local_optim.zero_grad()
                    _loss = _loss.mean()
                    _loss.backward()
                    # accumulate loss to glb_policy
                    for lp, gp in zip(
                        agent.local_policy.parameters(),
                        self.glb_policy.parameters(),
                    ):
                        gp._grad += lp.grad / self.num_agents

                    # add to glb_ratios & glb_advantages
                    # glb_ratios[rank_mask != agent.rank] += _ratios
                    # glb_advantages[rank_mask != agent.rank] += _advantages
                # glb_surr1 += _ratios * _advantages
                # glb_surr2 += torch.clamp(_ratios, 1 - self.eps_clip,
                #     1 + self.eps_clip) * _advantages

                # for _ in range(self.optim_epochs):
            # upload global policy
            # glb_ratios /= self.num_agents
            # glb_advantages /= self.num_agents
            # log_probs, state_values, dist_entropy = self.glb_policy.evaluate(
            #     old_states, old_actions)
            # glb_ratios = torch.exp(log_probs - old_logprobs.detach())
            # glb_surr1 = glb_ratios * glb_advantages
            # glb_surr2 = torch.clamp(glb_ratios, 1 - self.eps_clip,
            #     1 + self.eps_clip) * glb_advantages

            # loss = -torch.min(glb_surr1, glb_surr2) + 0.5 * F.mse_loss(state_values,
            #     rewards) - 0.01 * dist_entropy

            # take gradient step
            # self.glb_optimizer.zero_grad()
            # loss = loss.mean()
            # loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.glb_policy.parameters(),
                    self.grad_clip,
                )
                    # self.grad_clip * self.optim_epochs / self.num_agents)
            self.glb_optimizer.step()

        # del _glb_policy_state
        # del _glb_optim_state
        del list_local_optim
        for agent in self.agent_list:
            if agent.done: continue # no need to update for a done agent
            agent.update_local_policy()
            agent.reset_memory()


    def learn(self):
        # initialize
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                filename_suffix='_{}'.format(self.run_name),)
        self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.spawn_npc_vehicles(51 - self.num_agents)
        self.agent_list = [_PPO_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()

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
            self.glb_env.step()
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
                    self.recorder['recent']['avg_dist_to_trgt'].record_value(
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
                    self.tbwriter.add_scalar('recent/avg_dist_to_target',
                        self.recorder['recent']['avg_dist_to_trgt'].summary(),
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

            if self.num_steps_since_update >= self.glb_update_freq:
                self.num_steps_since_update = 0
                if self.resumed:
                    # skip the first update after resume
                    self.resumed = False
                else:
                    # do the learning
                    # print('updating policy...')
                    if self.nesterov:
                        self._nesterov_update()
                    else:
                        self._update()


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
                        glb_policy=self.glb_policy, rank=rk,
                        memory=self.agent_list[rk].memory)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()

    def test(self, videos=False):
        # assert self.num_agents == 1, '{} != 1'.format(self.num_agents)
        # initialize
        term_stats = defaultdict(int)
        if videos: viz = Visualizer(images_path=self.vid_log_dir,
            video_path=self.vid_log_dir)
        idx_list = list(range(self.num_agents))
        self.glb_num_test_episodes = self.glb_env.config['num_episodes']
        # print(self.glb_num_test_episodes)
        self.glb_env.reset(rank_list=self.rank_list, use_idx=True,
            idx_list=idx_list, reset_npc=True)
        self.glb_env.spawn_npc_vehicles()
        self.agent_list = [_PPO_Individual_Agent(
            self.glb_env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list]
        self.glb_env.reset_vehicle_agent(self.agent_list)
        self.glb_env.step()

        self.num_successes = 0
        while self.glb_num_episodes < self.glb_num_test_episodes + 1:
            # take action
            for rk, agent in enumerate(self.agent_list):
                # prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                action, _ = agent.select_action(deterministic=True)
                agent.action = action
            self.vprint('action chosen:', [a.action for a in self.agent_list])
            # get new observation
            self.glb_env.step()

            for rk, agent in enumerate(self.agent_list):
                if videos:
                    sub_folder='ep{}rk{}'.format(self.glb_num_episodes, rk)
                    viz.save_image(agent.rv_image, sub_folder=sub_folder)
                if agent.done:  # done and print information
                    term_stats[agent.termination_state] += 1
                    if videos:
                        viz.generate_video(sub_folder,
                            suffix=agent.termination_state)
                        viz.remove_images(sub_folder)
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
                    self.agent_list[rk] = _PPO_Individual_Agent(
                        self.glb_env.ego_vehicle_list[rk],
                        glb_policy=self.glb_policy, rank=rk, memory=None)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()
        print('[Finished]', term_stats)

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


class DPPO_Collective_Agent(object):
    def __init__(self, glb_env_list, glb_policy, glb_optimizer, num_agents=1,
        max_glb_num_steps=1000000, gamma=.99, eps_clip=.2, nesterov=False,
        glb_update_freq=1000, optim_epochs=100, focal_loss=False,
        grad_clip=None, save_freq=100000, save_suffix='', verbose=False):
        """A DPPO agent.
        Args:
            glb_env_list: global environment list
            glb_policy: network shared by all PPO agents
            glb_optimizer: optimizer for the glb_policy
            num_agents: list of numbers of PPO agents
            max_glb_num_steps: max number of global steps
            gamma: reward discount factor
            eps_clip: clip parameter for PPO
            grad_clip: value for clipping gradient, None to disable
            nesterov: if using nesterov update
            glb_update_freq: update frequency of glb_policy
            optim_epochs: update policy for how many epochs
            save_freq: checkpoint saving frequency
                (save the agent every N global steps)
            save_suffix: checkpoint saving suffix
            deterministic: evaluation mode (deterministic action)
            verbose: if print some debug information
        """
        super().__init__()
        self.glb_env_list = glb_env_list
        self.glb_policy = glb_policy
        self.policy_lock = Lock()
        self.glb_optimizer = glb_optimizer
        self.max_glb_num_steps = max_glb_num_steps
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.glb_update_freq = glb_update_freq
        self.optim_epochs = optim_epochs
        self.focal_loss = focal_loss
        self.num_agents = num_agents
        self.rank_list = [list(range(num_agents)) for _ in glb_env_list]
        self.agent_list = [[None] * num_agents for _ in glb_env_list]
        self.grad_clip = grad_clip
        self.nesterov = nesterov
        self.device = next(glb_policy.parameters()).device
        self.save_freq = save_freq
        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'DPPO{}x{}{}'.format(len(glb_env_list), num_agents,
            self.save_suffix)
        self.verbose = verbose
        self.glb_ep_reward_list = []
        self.agent_reward_list = \
            [[[] for _ in range(num_agents)] for _ in glb_env_list]
        self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.savetime = lambda: time.strftime('%b%d%I%M%p%S')
        self.tb_log_dir = '{}/{}_{}'.format('./tensorboard_logs',
            self.run_name, self.savetime())
        self.vid_log_dir = '{}/{}_{}'.format('./video_logs',
            'PPO_test', self.savetime())
        # self.glb_num_episodes = mp.Value('i', 1)
        self.glb_num_episodes = 1
        self.episode_lock = Lock()
        # self.glb_num_steps = mp.Value('i', 0)
        self.glb_num_steps = 0
        self.step_lock = Lock()
        # self.num_steps_since_update = mp.Value('i', 0)
        self.num_steps_since_update = 0
        self.update_lock = Lock()
        self.memory_cond = Condition()
        self.update_bar = Barrier(len(glb_env_list), timeout=None)
        self.recorder = GlobalRecorder
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

    def _update(self):
        # with self.memory_cond:
        #     self.memory_cond.wait()
        rewards = []
        old_states = []
        old_actions = []
        old_logprobs = []
        for env_id, _ in enumerate(self.glb_env_list):
            for agent in self.agent_list[env_id]:
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
        for _ in range(self.optim_epochs):
            # Evaluating old actions and values:
            # print(old_states.shape)
            logprobs, state_values, dist_entropy = self.glb_policy.evaluate(
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

            # take gradient step
            self.glb_optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.glb_policy.parameters(), self.grad_clip)
            self.glb_optimizer.step()

        for env_id, _ in enumerate(self.glb_env_list):
            for agent in self.agent_list[env_id]:
                if agent.done: continue # no need to update for a done agent
                agent.update_local_policy()
                agent.reset_memory()

    def _nesterov_update(self):
        raise NotImplementedError

    def _learn(self, env_id):
        # initialize
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                filename_suffix='_{}'.format(self.run_name),)
        env = self.glb_env_list[env_id]
        env.reset(rank_list=self.rank_list[env_id])
        env.spawn_npc_vehicles(51 - self.num_agents)
        self.agent_list[env_id] = [_PPO_Individual_Agent(
            env.ego_vehicle_list[i],
            glb_policy=self.glb_policy, rank=i) for i in self.rank_list[env_id]]
        env.reset_vehicle_agent(self.agent_list[env_id])
        env.step()

        avg_t_action, avg_t_step  = [], []

        while self.glb_num_steps < self.max_glb_num_steps + 1:
            # take action
            ts_action = time.time()
            for rk, agent in enumerate(self.agent_list[env_id]):
                # prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                action, logprob = agent.select_action()
                agent.action = action
                # update partial memory
                # with self.update_lock:
                agent.memory['state'].append(agent.observation.tolist())
                agent.memory['action'].append(action.tolist())
                agent.memory['logprob'].append(logprob.tolist())
            te_action = time.time()
            self.vprint('action chosen:', [a.action for a in self.agent_list[env_id]])
            # get new observation
            ts_step = time.time()
            env.step()
            te_step = time.time()
            avg_t_action.append(te_action - ts_action)
            avg_t_step.append(te_step - ts_step)
            self.vprint('[num_agent {}][action time {:.4f}, avg {:.4f}]'
                '[step time {:.4f}, avg {:.4f}]'.format(self.num_agents,
                avg_t_action[-1], np.mean(avg_t_action), avg_t_step[-1],
                np.mean(avg_t_step)))

            for rk, agent in enumerate(self.agent_list[env_id]):
                # with self.update_lock:
                agent.memory['reward'].append(agent.curr_reward)
                agent.memory['done'].append(agent.done)

                if agent.done:  # done and print information
                    print('[{}]'.format(self.time()) + \
                        '[{}]'.format(self.run_name) + \
                        '[glb ep {}][glb step {}][env {}, agent {}] done({})'
                        ', ep reward [{:.4f}]'.format(
                        self.glb_num_episodes, self.glb_num_steps,
                        env_id, rk, agent.termination_state,
                        agent.episode_reward))
                    self.agent_reward_list[env_id][rk].append(agent.episode_reward)
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
                    self.tbwriter.add_scalar('recent/avg_dist_to_target',
                        self.recorder['recent']['avg_dist_to_trgt'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/success_rate',
                        self.recorder['recent']['success_rate'].summary(),
                        self.glb_num_episodes)
                    self.tbwriter.add_scalar('recent/collision_rate',
                        self.recorder['recent']['collision_rate'].summary(),
                        self.glb_num_episodes)
                    self.recorder.summary_all()
                    with self.episode_lock:
                        self.glb_num_episodes += 1


                agent.num_total_steps += 1
                with self.step_lock:
                    self.num_steps_since_update += 1
                    self.glb_num_steps += 1

                # save checkpoint
                if env_id == 0 and self.glb_num_steps % self.save_freq == 0:
                    self.save()

            # with self.memory_cond:
            #     self.memory_cond.notify_all()

            # with self.update_lock:
            if self.num_steps_since_update >= self.glb_update_freq:
                self.update_bar.wait()
                if not self.resumed and env_id == 0:
                    # do the learning
                    # print('updating policy...')
                    if self.nesterov:
                        self._nesterov_update()
                    else:
                        self._update()
                self.update_bar.wait()
                with self.step_lock:
                    self.num_steps_since_update = 0
                    self.resumed = False


            # respawn dead agents
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list[env_id]):
                if agent.done: respawn_rank_list.append(rk)
            if len(respawn_rank_list) > 0: # there're dead agents to respawn
                env.reset(rank_list=respawn_rank_list)
                # update agent list
                for rk in respawn_rank_list:
                    self.agent_list[env_id][rk] = _PPO_Individual_Agent(
                        env.ego_vehicle_list[rk],
                        glb_policy=self.glb_policy, rank=rk,
                        memory=self.agent_list[env_id][rk].memory)
                env.reset_vehicle_agent(
                    [self.agent_list[env_id][rk] for rk in respawn_rank_list])
                env.step()

    def learn(self):
        # proc_list = []
       #  for env_id, _ in enumerate(self.glb_env_list):
       #      p = mp.Process(target=self._learn, args=(env_id,))
       #      proc_list.append(p)

       #  for p in proc_list:
       #      p.start()
       #  for p in proc_list:
       #      p.join()
       #
        thread_list = []
        for env_id, _ in enumerate(self.glb_env_list):
            p = Thread(target=self._learn, args=(env_id,))
            thread_list.append(p)

        for p in thread_list:
            p.start()
        for p in thread_list:
            p.join()

        print('Training Finished')

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
        raise NotImplementedError



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
