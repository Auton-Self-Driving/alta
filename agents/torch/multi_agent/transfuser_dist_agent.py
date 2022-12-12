import os
import time
import pickle
import copy
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import dist_utils as dist

from threading import Thread, Lock
from collections import Counter, OrderedDict
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


class _Dist_Individual_Agent(Agent):
    def __init__(self, vehicle, glb_policy, timestamp=-1,
        rank=None, memory=None, **kwargs):
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
        self.timestamp = timestamp

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


class Dist_Worker_Agent(object):
    def __init__(self, local_env, local_policy,
        num_agents=1, save_suffix='', verbose=False):
        """A Distributed Data Collecting Worker agent.
        Args:
            local_env: the global environment
            local_policy: network shared by this local environment
            num_agents: number of policy controlled agent
            save_suffix: checkpoint saving suffix
            verbose: if print some debug information
        """
        self.local_env = local_env
        self.local_policy = local_policy
        self.N_S = self.local_policy.N_S
        self.N_A = self.local_policy.N_A
        self.model_len = len(parameters_to_vector(local_policy.parameters()))
       
        self.rank_list = list(range(num_agents))
        self.agent_reward_list = [[] for _ in self.rank_list]
        self.glb_ep_reward_list = []
        self.res_queue = [[] for _ in self.rank_list]
        self.agent_list = None
        self.verbose = verbose

        ################################################################
        
        self.init_dist_framework()
        self.device = next(local_policy.parameters()).device

        self.save_suffix = '_' + save_suffix if save_suffix else ''
        self.run_name = 'DPPO{}x{}x{}{}'.format(self.num_servers,
            self.num_workers, self.num_agents, self.save_suffix)
        self.tb_log_dir = '{}/{}'.format('./tensorboard_logs',
            self.run_name)

        self.glb_num_episodes = 1

        self.recorder = GlobalRecorder
        self.tbwriter = None
        self.resumed = False

    def init_dist_framework(self):
        comm_vars = dist.init_param_server_comm()
        self.rank, self.world_size = comm_vars[:2]
        self.server_list, self.worker_list = comm_vars[2:4]
        self.server_group, self.worker_group = comm_vars[4:]
        self.num_servers = len(self.server_list)
        self.num_workers = len(self.worker_list)
        self.server_rank = (self.rank - self.num_servers) % self.num_servers
        self.vprint(comm_vars, self.server_rank)
        self.recv_info_len = 3

    def vprint(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def to_tensor(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).to(self.device)

    def _update_buffer(self, agent):
        """ Sends collected trajectory to server and receives latest parameters
        """

        # Organize current episode to send to server
        mem = agent.memory
        vec_mem = np.array(mem['action']).flatten().tolist()
        vec_mem.extend(np.array(mem['state']).flatten().tolist())
        vec_mem.extend(np.array(mem['logprob']).flatten().tolist())
        vec_mem.extend(mem['reward'])
        vec_mem.extend(mem['done'])
        vec_mem = torch.tensor(vec_mem)
        
        # Create packet and notify server of immenent arrival of episode trajectory
        overhead = [self.rank, len(mem['reward']),
            len(mem['reward']), agent.timestamp, SIG.GRAD_PUSH]
        dist.isend(overhead, dst=self.server_rank, tag=SIG.QUERY).wait()

        # Sending episode trajectory over to server
        dist.isend(overhead, vec_mem, dst=self.server_rank, tag=SIG.GRAD).wait()

        # Purging episode from agent memory
        agent.reset_memory()

        # Updating local policy with latest parameters from server
        self.glb_num_steps, self.glb_num_episodes, \
            self.local_policy_timestamp = self.update_parameters()

    def _record_stats(self, agent, success_int, obs_collision_int):

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
        # self.recorder.summary_all() # For printing

    def learn(self):

        # Load Parameters
        self.load_agent()

        # Initialize tensorboard instance of worker
        if self.tbwriter is None:
            self.tbwriter = TensorboardWriter(
                log_dir=self.tb_log_dir,
                filename_suffix='_{}'.format(self.run_name),)

        # Initialize local environment
        self.local_env.reset(rank_list=self.rank_list) # Ranklist = [0,1,...,num_agents-1]
        self.local_env.spawn_npc_vehicles(51 - self.num_agents)
        # Add agents to local env that are controllable by the policy network
        self.agent_list = [_DPPO_Individual_Agent( 
            self.local_env.ego_vehicle_list[i], timestamp=0,
            glb_policy=self.local_policy, rank=i) for i in self.rank_list]
        self.local_env.reset_vehicle_agent(self.agent_list, transfuser=False)
        self.curr_town = self.local_env.curr_town
        self.local_env.step()

        avg_t_action, avg_t_step  = [], []

        while self.glb_num_steps < self.max_glb_num_steps + 1:
                    
            # Used to compute time taken to take action
            ts_action = time.time() 

            # Select an action for all controllable agents
            for rk, agent in enumerate(self.agent_list):

                # Choose agent action based on its state
                action, logprob = agent.select_action()

                agent.action = action
                agent.memory['state'].append(agent.observation.tolist())
                agent.memory['action'].append(action.tolist())
                agent.memory['logprob'].append(logprob.tolist())
            te_action = time.time()
            self.vprint('action chosen:', [a.action for a in self.agent_list])

            # Perform 1 step of the environment
            ts_step = time.time()
            self.local_env.step()
            te_step = time.time()

            # Store time taken to perform an action
            avg_t_action.append(te_action - ts_action)
            # Store time taken to perform an env step (generally 10x of action time)
            avg_t_step.append(te_step - ts_step)

            self.vprint('[num_agent {}][action time {:.4f}, avg {:.4f}]'
                '[step time {:.4f}, avg {:.4f}]'.format(self.num_agents,
                avg_t_action[-1], np.mean(avg_t_action), avg_t_step[-1],
                np.mean(avg_t_step))) 

            # Collect agent rewards, update statistics and send episode 
            # to server if complete
            for rk, agent in enumerate(self.agent_list):

                agent.memory['reward'].append(agent.curr_reward)
                agent.memory['done'].append(agent.done)

                agent.num_total_steps += 1
                self.num_steps_since_update += 1
                self.local_num_steps += 1

                # Send trajectory to server if max rollout length achieved
                if not self.push_grad and len(agent.memory['done']) >= self.grad_update_freq:
                    self._update_buffer(agent)

                if agent.done:  # done and print information

                    # print('[{}]'.format(self.time()) + \
                    #     '[{}][rank {}]'.format(self.run_name, self.rank) + \
                    #     '[local ep {}][local step {}][agent {}] done({})'
                    #     ', ep reward [{:.4f}]'.format(
                    #     self.local_num_episodes, self.local_num_steps, rk,
                    #     agent.termination_state, agent.episode_reward))

                    self.agent_reward_list[rk].append(agent.episode_reward)
                    self.glb_ep_reward_list.append(agent.episode_reward)

                    success_int = int('success' == agent.termination_state)
                    obs_collision_int = int('obs_collision' == agent.termination_state)
                    self._record_stats(agent, success_int, obs_collision_int)
                    self.local_num_episodes += 1
                    self.num_eps_since_update += 1

                    # Push Trajectory to server on episode completion
                    if not self.standard and len(agent.memory['done']) > 0:
                        self._update_buffer(agent)
                        self.num_steps_since_update = 0
                        self.num_eps_since_update = 0


            # Identify dead agents
            respawn_rank_list = []
            for rk, agent in enumerate(self.agent_list):
                if agent.done: respawn_rank_list.append(rk)
  
            # Execute Respawning if dead agents present
            if len(respawn_rank_list) > 0: 

                self.local_env.reset(rank_list=respawn_rank_list)

                # If environment town has changed, respawn all agents 
                # in the worker in the new town
                if self.curr_town != self.local_env.curr_town:

                    self.curr_town = self.local_env.curr_town
                    self.local_env.reset(rank_list=self.rank_list)

                    for rk in self.rank_list:
                        self.agent_list[rk] = _DPPO_Individual_Agent(
                            self.local_env.ego_vehicle_list[rk],
                            glb_policy=self.local_policy,
                            timestamp=self.local_policy_timestamp,
                            rank=rk, memory=None)

                    self.local_env.reset_vehicle_agent(
                        [self.agent_list[rk] for rk in self.rank_list], transfuser=False)
                    self.local_env.scenario_index = 0

                else:
                    # Respawn a dead agent
                    for rk in respawn_rank_list:
                        self.agent_list[rk] = _DPPO_Individual_Agent(
                            self.local_env.ego_vehicle_list[rk],
                            glb_policy=self.local_policy,
                            timestamp=self.local_policy_timestamp,
                            rank=rk, memory=self.agent_list[rk].memory)
                    
                    # Reset vehicle of dead agent in environment
                    self.local_env.reset_vehicle_agent(
                        [self.agent_list[rk] for rk in respawn_rank_list],  transfuser=False)

                # QUESTION - why step?
                self.local_env.step()

    def load_agent(self):
        self.local_policy.load_state_dict(checkpoint['local_policy'])
        print('checkpoint params loaded')