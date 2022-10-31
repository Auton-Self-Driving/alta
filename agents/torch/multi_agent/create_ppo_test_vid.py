import sys
import os
import glob
import traceback
CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

import carla
import numpy as np
import torch
import gym
from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG, DPPO_CONFIG
from dppo_agent import DPPO_Server_Agent, DPPO_Worker_Agent, _DPPO_Individual_Agent
import time
import cv2

attr_of_int_agent = ['_vehicle','_world','action','actor_list','image_data','prev_measurement','rv_camera_actor','rv_image','vehicle_actor']
attr_of_int_vehicle = ['get_location','get_transform']

def env_reset_wrapper(env, policy):
    rank_list = [0]
    env.reset(rank_list=rank_list)
    env.spawn_npc_vehicles(51 - len(rank_list))
    agent_list = [_DPPO_Individual_Agent(
        env.ego_vehicle_list[i], timestamp=0,
        glb_policy=policy, rank=i) for i in rank_list]
    env.reset_vehicle_agent(agent_list)
    curr_town = env.curr_town
    env.step()
    return agent_list

    # for rk, agent in enumerate(env.ego_agent_list):
    #     if agent.action is None:
    #         # obs = env._get_ego_input(agent)
    #         spectator = env._world.get_spectator()
    #         #ego_vehicle._vehicle.get_transform()
    #         # print(obs)
    #         print('\n\n\n')
    #         print(agent.prev_measurement['episode_id'],agent.prev_measurement['location'],agent.prev_measurement['num_steps'])
    #         print('\n\n\n')
    #         print("Agent Attributes:")
    #         for attr in attr_of_int_agent:
    #             print('{} : {}'.format(attr,agent.__dict__[attr]))

    #         print('{} : {}'.format('actor_list[0]',dir(agent.actor_list[0])))
    #         print('\n\n\n')
    #         print("Vehicle Agent Attributes:")
    #         print(vars(agent.vehicle_actor))
    #         # for attr in attr_of_int_vehicle:
    #         #     print('{} : {}'.format(attr,agent.vehicle_actor.__dict__[attr]))
    # exit()

def get_state_action_dims_from_config(ENV_CONFIG):
    if ENV_CONFIG['input_type'] == 'wp_obs_info_speed_steer_ldist_goal_light':
        N_S, N_A = 8, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_info_speed_steer_ldist_light':
        N_S, N_A = 7, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
        N_S, N_A = 11, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light':
        N_S, N_A = 15, 2
    elif ENV_CONFIG['input_type'] == 'wp_360_obstacle_speed_steer':
        N_S, N_A = 24, 2
    else:
        N_S, N_A = 7, 2

    return N_S, N_A


def main():
    
    device = ENV_CONFIG['device']

    N_S, N_A = get_state_action_dims_from_config(ENV_CONFIG)
    
    env = CarlaEnv(config = ENV_CONFIG)
    policy = PPOActorCritic_Continuous(N_S, N_A,
        use_transformer=ENV_CONFIG['input_type']=='transformer') # global network

    if DPPO_CONFIG['checkpoint']:
        ckpt = torch.load(DPPO_CONFIG['checkpoint'], map_location='cpu')
        policy.load_state_dict(ckpt['glb_policy'])

    # policy = PPO.load('checkpoints/policy_checkpoint__21300_steps', device = 0)

    try:
        #25 routes for Town 1
        episodes = 25
        success = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for i in range(episodes):
            agent_list = env_reset_wrapper(env, policy)
            done = False
            # video=cv2.VideoWriter('videos/'+str(i)+'.mp4',fourcc,10,(512,512))
            itr = 0
            max_iter = 500
            while(not done):

                for rk, agent in enumerate(agent_list):
                    # prev_obs = torch.from_numpy(agent.observation).to(torch.float)
                    action, logprob = agent.select_action()
                    agent.action = action
                    # update partial memory
                    agent.memory['state'].append(agent.observation.tolist())
                    agent.memory['action'].append(action.tolist())
                    agent.memory['logprob'].append(logprob.tolist())

                    done = agent.done

                itr += 1
                if itr > max_iter:
                    break

                env.step()

            print(agent.memory['state'][0])
            break

                # img = info["sensor.camera.rgb/top"]
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     video.write(img)            
            # video.release()
            # print(info)
            # success += int(info['termination_state'] == "success")
            # print('SUCCESS RATE: ',success,'/',i+1)
    except Exception:
        traceback.print_exc()
        env.close()

if __name__ == '__main__':
    main()

# state_tensor = torch.from_numpy(obs).to(torch.float)#.to(self.device)
# action, logprob = policy.act(state_tensor,deterministic=True)
# obs, reward, done, info = env.step(action)
