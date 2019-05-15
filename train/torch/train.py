from tensorboardX import SummaryWriter
import numpy as np
import random
import argparse
import os

import torch
import torch.nn as nn
from torch.distributions import Normal
from environment.mujoco import MujocoEnv

from agents.torch import DDPGAgent, TD3Agent, SACAgent, ReplayMemory, OUNoise
from agents.torch.networks import *

from utils import *


ALLOWED_ENVS = ["Carla-8-2", "Carla-9-4", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2", 
                "InvertedPendulum-v2", "InvertedDoublePendulum-v2", "Reacher-v2", "CarRacing-v0", 
                "Pong-v0"]
ALLOWED_ALGOS = ["DDPG", "TD3", "SAC"]


def make_env_and_agent(args):
    assert args.env_name in ALLOWED_ENVS
    assert args.algo in ALLOWED_ALGOS
    
    if args.action_type == "sep_gas":
        action_dim = 3
    elif args.action_type == "merged_gas":
        action_dim = 2
    elif args.action_type == "steer_only":
        action_dim = 1

    if args.algo == "DDPG":
        if "Carla" in args.env_name:
            if args.env_name == "Carla-8-2":
                from environment.carla_8_2 import CarlaEnv as CarlaEnv82
                
                print("Using Carla 0.8.2 version environment")
                env = CarlaEnv82(port=args.carla_port)
            elif args.env_name == "Carla-9-4":
                # from environment.carla_9_4.env import CarlaEnv as CarlaEnv94
                # from environment.carla_9_4.config import ConfigManager
                
                # print("Using Carla 0.9.4 version environment")
                # config = ConfigManager(
                #     algo=args.algo, action_type=args.action_type, reward=args.reward_function)
                # env = CarlaEnv94(config=config.config, port=args.carla_port)
                from environment.carla_9_4.environment import CarlaEnv as CarlaEnv94
                from environment.carla_9_4.config import ConfigManager
                print("Using Carla 0.9.4 version environment")
                config = ConfigManager(
                    algo=args.algo, action_type=args.action_type, reward=args.reward_function, use_segmented=args.segmented)
                env = CarlaEnv94(config=config.config, port=args.carla_port)
            
            if args.pretrained == "none":
                if args.cnn_size == "Large":
                    cnn = DrivingLargeCNN(args.batch_norm, args.dropout_rate)
                elif args.cnn_size == "Small":
                    cnn = DrivingSmallCNN(args.batch_norm, args.dropout_rate)
                elif args.cnn_size == "Smallest":
                    cnn = DrivingSmallestCNN(args.batch_norm, args.dropout_rate)
            else:
                cnn = Pretrained(model_name=args.pretrained, pre_trained=True)
            print(cnn)
            actor = DrivingDeterministicPolicy(action_dim=action_dim)
            critic = DrivingQValue(action_dim=action_dim)
            
            noise = OUNoise(mu=torch.zeros(action_dim),
                            sigma=args.sigma_noise * torch.ones(action_dim),
                            theta=args.theta_noise * torch.ones(action_dim))
            
        else:
            env = MujocoEnv(args.env_name, pixel_obs=args.pixel_obs)
            
            if args.pixel_obs:
                cnn = MujocoSmallCNN(args.batch_norm)
                actor = MujocoDeterministicPolicy(400, env.action_dim, 
                                                  pixel_obs=args.pixel_obs)
                critic = MujocoQValue(400, env.action_dim, pixel_obs=args.pixel_obs)
            
            else:
                cnn = None
                actor = MujocoDeterministicPolicy(env.state_dim, env.action_dim)
                critic = MujocoQValue(env.state_dim, env.action_dim)
                
            noise = Normal(torch.zeros(env.action_dim), 
                           torch.ones(env.action_dim) * 0.1)
            
        agent = DDPGAgent(cnn, actor, critic, noise, args.actor_lr, args.critic_lr,
                          args.target_lr, args.discount, args.max_grad_norm)
    
    elif args.algo == "TD3":
        
        if "Carla" in args.env_name:
            if args.env_name == "Carla-8-2":
                from environment.carla_8_2 import CarlaEnv as CarlaEnv82
                
                print("Using Carla 0.8.2 version environment")
                env = CarlaEnv82(port=args.carla_port)
            elif args.env_name == "Carla-9-4":
                from environment.carla_9_4.env import CarlaEnv as CarlaEnv94
                from environment.carla_9_4.config import ConfigManager

                print("Using Carla 0.9.4 version environment")
                config = ConfigManager(algo=args.algo)
                env = CarlaEnv94(config=config.config, port=args.carla_port)
            
            cnn = DrivingSmallCNN(args.batch_norm)
            actor = DrivingDeterministicPolicy()
            critic1 = DrivingQValue()
            critic2 = DrivingQValue()
            
            exploration_noise = OUNoise(mu=torch.zeros(2),
                                        sigma=args.sigma_noise * torch.ones(2),
                                        theta=args.theta_noise * torch.ones(2))
            target_noise = Normal(torch.zeros(2),
                                  args.sigma_noise * torch.ones(2))
            noise_clip = 0. # No target noise
            
        else:
            env = MujocoEnv(args.env_name, pixel_obs=args.pixel_obs)
            
            if args.pixel_obs:
                cnn = MujocoSmallCNN(args.batch_norm)
                actor = MujocoDeterministicPolicy(400, env.action_dim,
                                                  pixel_obs=args.pixel_obs)
                critic1 = MujocoQValue(400, env.action_dim, pixel_obs=args.pixel_obs)
                critic2 = MujocoQValue(400, env.action_dim, pixel_obs=args.pixel_obs)
                
            else:
                cnn = None
                actor = MujocoDeterministicPolicy(env.state_dim, env.action_dim)
                critic1 = MujocoQValue(env.state_dim, env.action_dim)
                critic2 = MujocoQValue(env.state_dim, env.action_dim)
                
            exploration_noise = Normal(torch.zeros(env.action_dim),
                                       torch.ones(env.action_dim) * 0.1)
            target_noise = Normal(torch.zeros(env.action_dim),
                                  torch.ones(env.action_dim) * 0.2)
            noise_clip = 0. # No target noise
        
        agent = TD3Agent(cnn, actor, critic1, critic2, exploration_noise,
                         target_noise, noise_clip, args.policy_interval, 
                         args.actor_lr, args.critic_lr, args.target_lr, 
                         args.discount, args.max_grad_norm)
        
    elif args.algo == "SAC":
        
        if args.env_name == "Carla-8-2":
            raise NotImplementedError
            
        else:
            
            if args.pixel_obs:
                raise NotImplementedError
                
            else:
                env = MujocoEnv(args.env_name)
                cnn = None
                actor = MujocoTanhGaussianPolicy(env.state_dim, env.action_dim)
                qcritic1 = MujocoQValue(env.state_dim, env.action_dim)
                qcritic2 = MujocoQValue(env.state_dim, env.action_dim)
                vcritic = MujocoStateValue(env.state_dim)
        
        agent = SACAgent(cnn, actor, qcritic1, qcritic2, vcritic, args.entropy_coeff, 
                         args.policy_l2, args.policy_interval, args.actor_lr,
                         args.critic_lr, args.target_lr, args.discount)
       
    return env, agent


    
def sample_and_train(args):
    dict_keys = ['collision_reward', 'distance_reward', 'speed_reward', 'control_throttle', 'speed', 
    'control_hand_brake', 'distance_to_goal', 'reward', 'control_steer', 'total_reward', 'control_brake', 
    'lane_intersection_reward', 'control_reverse']
    def train(stop_environment=False):
        # Train with a batch of transitions
        batch = memory.sample(args.batch_size)
        losses, weights, grads = agent.update(batch)

        # Record training statistics
        if total_steps % args.log_interval == 0:
            print(total_steps)

            for k, v in losses.items():
                writer.add_scalar('losses/' + k, v, total_steps)

            # for k, v in weights.items():
            #     #writer.add_histogram('weights/' + k, v, total_steps)
            #     writer.add_scalar(
            #         'weight_norms/' + k, np.linalg.norm(v), total_steps)

            # for k, v in grads.items():
            #     #writer.add_histogram('grads/' + k, v, total_steps)
            #     writer.add_scalar(
            #         'grad_norms/' + k, np.linalg.norm(v), total_steps)
            if not stop_environment:
                if args.env_name == "Carla-9-4":
                    for key in dict_keys:
                        if key in step_info:
                            writer.add_scalar('step/' + key, step_info[key], total_steps)
                elif args.env_name == "Carla-8-2":
                    for k, v in step_info.items():
                        writer.add_scalar('step/' + k, v, total_steps)
        if done and not stop_environment:
            print(total_steps)
            writer.add_scalar('episodes/train/reward', episode_reward, total_steps)
            writer.add_scalar('episodes/train/length', episode_steps, total_steps)
            if "Carla" in args.env_name:
                writer.add_scalar(
                    'episodes/train/dist_to_target', obs["dist_to_target"].item(), total_steps)
                writer.add_scalar('episodes/train/success',
                                success_episodes, total_steps)
                writer.add_scalar('episodes/train/collision',
                                collision_episodes, total_steps)
                writer.add_scalar('episodes/train/offlane',
                                offlane_episodes, total_steps)
                writer.add_scalar('episodes/train/static',
                                static_episodes, total_steps)
                writer.add_scalar('episodes/train/max_steps',
                                max_steps_episodes, total_steps)

        # Save weights
        if total_steps % args.save_interval == 0:
            agent.save(save_file)
    
    def validate():
        # Define as non local variables to update local copy
        nonlocal success_val_episodes
        nonlocal collision_val_episodes
        nonlocal offlane_val_episodes
        nonlocal static_val_episodes
        nonlocal max_steps_val_episodes

        obs = env.reset()
        if args.env_name == "Carla-9-4":
            obs = convert_observation(obs)
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            action, _ = agent.get_action(obs, eval_mode=True)

            if args.env_name == "Carla-9-4":
                next_obs, reward, done, step_info = env.step(
                    to_numpy(action))
                next_obs = convert_observation(next_obs)
                reward = from_numpy(reward)
                done = from_numpy(done)
            else:
                next_obs, reward, done, step_info = env.step(action)

            episode_steps += 1
            episode_reward += reward.item()

            obs = next_obs

            # Update global episode termination state count for Carla
            if "termination_state" in step_info:
                termination_state = step_info["termination_state"]
                if termination_state is 'success':
                    success_val_episodes += 1
                elif termination_state is 'collision':
                    collision_val_episodes += 1
                elif termination_state is 'offlane':
                    offlane_val_episodes += 1
                elif termination_state is 'static':
                    static_val_episodes += 1
                elif termination_state is 'max_steps':
                    max_steps_val_episodes += 1
                del step_info["termination_state"]

        writer.add_scalar('episodes/val/reward', episode_reward, total_steps)
        writer.add_scalar('episodes/val/length', episode_steps, total_steps)
        if "Carla" in args.env_name:
            writer.add_scalar(
                'episodes/val/dist_to_target', obs["dist_to_target"].item(), total_steps)
            writer.add_scalar('episodes/val/success',
                            success_val_episodes, total_steps)
            writer.add_scalar('episodes/val/collision',
                            collision_val_episodes, total_steps)
            writer.add_scalar('episodes/val/offlane',
                            offlane_val_episodes, total_steps)
            writer.add_scalar('episodes/val/static',
                            static_val_episodes, total_steps)
            writer.add_scalar('episodes/val/max_steps',
                            max_steps_val_episodes, total_steps)

    # Ensure log_dir and save_dir are present
    silent_add(args.log_dir, args.save_dir)

    # Remove log and save files if they exist
    log_file = os.path.join(args.log_dir, args.file_name)
    save_file = os.path.join(args.save_dir, args.file_name)
    silent_remove(log_file, save_file)
    
    # Fix random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Tensorboard summary writer
    writer = SummaryWriter(log_file)
    
    # Environment and agent
    env, agent = make_env_and_agent(args)
    
    # Load weights learned previously
    if args.load_file is not None:
        agent.load(args.load_file)
    
    # Replay buffer
    memory = ReplayMemory(args.replay_size)
        
    total_steps = 0
    val_steps = 0
    val_step_interval = args.val_steps

    # Episode termination state count for Carla
    success_episodes = 0
    collision_episodes = 0
    offlane_episodes = 0
    static_episodes = 0
    max_steps_episodes = 0

    # Validation Episode termination state count for Carla
    success_val_episodes = 0
    collision_val_episodes = 0
    offlane_val_episodes = 0
    static_val_episodes = 0
    max_steps_val_episodes = 0
    
    while total_steps < args.max_steps:
        if total_steps > val_steps:
            print("Validating DDPG networks!!!")
            validate()
            val_steps += val_step_interval

        # Start new episode
        if not args.fixed_replay or total_steps <= args.replay_size:
            obs = env.reset()
            if args.env_name == "Carla-9-4":
                obs = convert_observation(obs)
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done:
                # Take a random action to bootstrap exploration
                if total_steps < args.start_steps:
                    if "Carla" in args.env_name:
                        if args.action_type == "merged_gas":
                            action = torch.tensor([[0., 1.]])
                        elif args.action_type == "steer_only":
                            action = torch.tensor([[0.]])
                        
                        action += agent.noise.sample()
                    else:
                        action = torch.zeros(1, env.action_dim).uniform_(-1, 1)
                    
                # Or compute action 
                else:
                    action, _ = agent.get_action(obs, eval_mode=True)
                    # action, _ = agent.get_action(obs, eval_mode=False) # Original
                
                
                # Take a step in environment    
                if args.env_name == "Carla-9-4":
                    next_obs, reward, done, step_info = env.step(
                        to_numpy(action))
                    next_obs = convert_observation(next_obs)
                    reward = from_numpy(reward)
                    done = from_numpy(done)
                else:
                    next_obs, reward, done, step_info = env.step(action)
                    
                total_steps += 1
                episode_steps += 1
                episode_reward += reward.item()

                # Update global episode termination state count for Carla
                if "termination_state" in step_info:
                    termination_state = step_info["termination_state"]
                    if termination_state is 'success':
                        success_episodes += 1
                    elif termination_state is 'collision':
                        collision_episodes += 1
                    elif termination_state is 'offlane':
                        offlane_episodes += 1
                    elif termination_state is 'static':
                        static_episodes += 1
                    elif termination_state is 'max_steps':
                        max_steps_episodes += 1
                    del step_info["termination_state"]
                
                # Push transition to memory
                memory.push((obs, action, reward, next_obs, done))
                
                obs = next_obs
                
                # Calling the inner train() function
                train()
        else:
            # Stop updating the replay buffer and stop sampling from environment.
            # Sample and train on elements from replay buffer
            # Calling the inner train() function
            train(stop_environment=True)
            total_steps += 1
        
                
if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env-name',
                        help='one of {}'.format(ALLOWED_ENVS))
    parser.add_argument('--algo', default='DDPG', 
                        help='one of {}'.format(ALLOWED_ALGOS))
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--discount', default=0.99, type=float,
                        help='discount factor')
    parser.add_argument('--actor-lr', default=1e-3, type=float,
                        help='actor learning rate')
    parser.add_argument('--critic-lr', default=1e-3, type=float,
                        help='critic learning rate')
    parser.add_argument('--target-lr', default=5e-3, type=float,
                        help='target networks learning rate')
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='training batch size')
    parser.add_argument('--replay-size', default=1000000, type=int,
                        help='size of replay buffer')
    parser.add_argument('--fixed-replay', action='store_true',
                        help='Fix replay buffer')
    parser.add_argument('--start-steps', default=0, type=int,
                        help='number of random steps to aid exploration')
    parser.add_argument('--val-steps', default=1000, type=int,
                        help='Validation step size')
    parser.add_argument('--max-steps', default=1e6, type=int,
                        help='number of environment steps to train')
    parser.add_argument('--max-grad-norm', default=10, type=float,
                        help='maximum norm for gradient clipping')
    
    # Environment specific parameters
    parser.add_argument('--carla-port', default='2000', type=int,
                        help='port for Carla server')
    parser.add_argument('--pixel-obs', action='store_true',
                        help='Mujoco observation in pixels if True')
    parser.add_argument('--pretrained', default="none",
                        help='one of "none", "vgg16", "resnet18", "squeezenet", "densenet", "inception"')
    parser.add_argument('--cnn-size', default="Small",
                        help='one of "Large", "Small", "Smallest"')
    parser.add_argument('--batch-norm', action='store_true',
                        help='apply batch normalization only if True')
    parser.add_argument('--segmented', action='store_true',
                        help='Use segmented image instead of RGB image')
    
    
    # Algorithm specific parameters
    parser.add_argument('--policy-interval', default=2, type=int,
                        help='number of critic updates per actor update - TD3, SAC')
    parser.add_argument('--sigma-noise', default=0.05, 
                        help='standard deviation of noise - DDPG, TD3')
    parser.add_argument('--theta-noise', default=0.15, 
                        help='mean revert of ounoise - DDPG, TD3')
    parser.add_argument('--entropy-coeff', default=0.2, 
                        help='entropy loss coefficient - SAC')
    parser.add_argument('--policy-l2', default=1e-3, 
                        help=' L2 regularization on policy parameters - SAC')
    parser.add_argument('--optim', default='Adam',
                        help='one of "Adam", "RMSprop"')
    parser.add_argument('--action-type', default='merged_gas',
                        help='one of "sep_gas", "merged_gas", "steer_only"')
    parser.add_argument('--reward-function', default='new',
                        help='one of "simplest", "new", "cirl" or "corl"')
    
    # Saving and logging parameters
    parser.add_argument('--log-dir', default='logs/', 
                        help='directory to save Tensorboard logs to')
    parser.add_argument('--save-dir', default='weights/', 
                        help='directory to save weight files to')
    parser.add_argument('--file-name', default="default",
                        help='file name inside log-dir and save-dir')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='log every log-interval updates')
    parser.add_argument('--save-interval', default=1000, 
                        help='save every save-interval updates')
    parser.add_argument('--load-file', default=None, 
                        help='file to load weight files from, if not None')
    
    args = parser.parse_args()
    sample_and_train(args)
