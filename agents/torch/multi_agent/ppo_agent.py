import os
import time
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from collections import deque
from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG
from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.dashcam import GlobalRecorder, TensorboardWriter

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
        # update partial memory
        self.memory['state'].append(prev_state.tolist())
        self.memory['action'].append(action.tolist())
        self.memory['logprob'].append(logprob.tolist())
        return action

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
    def __init__(self, glb_env, glb_policy, glb_optimizer,
        num_agents=1, max_glb_num_steps=1000000, gamma=.99, eps_clip=.2,
        glb_update_freq=1000, optim_epochs=100, save_freq=100000,
        save_suffix='', verbose=False):
        """An synchronous PPO agent.
        Args:
            glb_env: the global environment
            glb_policy: network shared by all PPO agents
            glb_optimizer: optimizer for the glb_policy
            num_agents: number of PPO agents
            max_glb_num_steps: max number of global steps
            gamma: reward discount factor
            eps_clip: clip parameter for PPO
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
        self.num_agents = num_agents
        self.rank_list = list(range(num_agents))
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        self.device = next(glb_policy.parameters()).device
        self.save_freq = save_freq
        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'PPOx{}{}'.format(self.num_agents,
            self.save_suffix)
        self.verbose = verbose
        self.glb_ep_reward_list = []
        self.agent_reward_list = [[] for _ in self.rank_list]
        self.time = lambda: time.strftime('%Y-%m-%d %H:%M:%S')
        self.glb_num_episodes = 1
        self.glb_num_steps = 0
        self.num_steps_since_update = 0
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
        old_states = torch.tensor(old_states, dtype=torch.float32, device=self.device).squeeze().detach()
        # print(old_states.shape)
        old_actions = torch.tensor(old_actions, dtype=torch.float32, device=self.device).squeeze().detach()
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device).squeeze().detach()

        # Optimize policy for K epochs:
        for _ in range(self.optim_epochs):
            # Evaluating old actions and values:
            # print(old_states.shape)
            logprobs, state_values, dist_entropy = self.glb_policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            # print('state_values', state_values.shape)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.glb_optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.glb_optimizer.step()

        for agent in self.agent_list:
            if agent.done: continue # no need to update for a done agent
            agent.update_local_policy()
            agent.reset_memory()

    def learn(self):
        # initialize
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir='./tensorboard_logs/',
                filename_suffix='_{}'.format(self.run_name),)
        self.glb_env.reset(rank_list=self.rank_list)
        self.glb_env.spawn_npc_vehicles()
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
                action = agent.select_action()
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
                    self.recorder['recent']['dist_to_target'].record_value(
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
                        self.recorder['recent']['mean_dist_to_target'].summary(),
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
                    self.agent_list[rk] = _PPO_Individual_Agent(
                        self.glb_env.ego_vehicle_list[rk],
                        glb_policy=self.glb_policy, rank=rk,
                        memory=self.agent_list[rk].memory)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()

    def test(self):
        # assert self.num_agents == 1, '{} != 1'.format(self.num_agents)
        # initialize
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
                    self.agent_list[rk] = _PPO_Individual_Agent(
                        self.glb_env.ego_vehicle_list[rk],
                        glb_policy=self.glb_policy, rank=rk, memory=None)
                self.glb_env.reset_vehicle_agent(
                    [self.agent_list[rk] for rk in respawn_rank_list])
                self.glb_env.step()

    def save(self, filename=None):
        if filename is None: filename = './ckpt{}_{}_{}.pth'.format(
            self.run_name, self.glb_num_steps, time.strftime('%b%d%I%M%p%S'))
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

    glb_policy = PPOActorCritic_Continuous(N_S, N_A) # global network

    print('*' * 80)
    print('FINISHED')
    print('*' * 80)
