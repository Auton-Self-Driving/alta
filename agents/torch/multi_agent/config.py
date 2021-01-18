"""Config for launching Multi-agent Carla Server
"""

import os
import sys
import glob

CARLA_9_4_PATH = os.environ.get('CARLA_9_4_PATH')
if CARLA_9_4_PATH == None:
    raise ValueError('Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh')

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('.egg file not found! Kindly check for your Carla installation.')
    pass


def get_discrete_actions():
    # steer = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    steer = [-0.3, -0.1, 0.0, 0.1, 0.3]
    target_speed = [0, 20]

    # Dictionary of discrete (Target_Speed, Steer) actions
    action_space = {}

    n = 0
    for i in range(len(target_speed)):
        for j in range(len(steer)):
            action_space[n] = [target_speed[i], steer[j]]
            n = n+1

    action_space[n] = [20, -0.5]
    action_space[n+1] = [20, 0.5]
    return action_space

DISCRETE_ACTIONS = get_discrete_actions()

EPISODE_MEASUREMENTS = {
    'episode_id': None,
    'num_steps': None,
    'location': None,
    'speed': None,
    'distance_to_goal': None,
    'num_collisions': 0,
    'num_laneintersections': 0,
    'static_steps': 0,
    'offlane_steps': 0,
    'control_steer': 0
}

SAC_CONFIG = {
    'policy_lr': 1e-3,
    'q_lr': 1e-3,
    'alpha_lr': 1e-4,
    'buffer_len': 100000,
    'target_entropy': -2.,
    'tau': .01,
    'batch_size': 512,
    'q_update_freq': 1,
    'target_update_freq': 1,
}

PPO_CONFIG = {
    'policy_lr': 1e-4,
    'eps_clip': .2,
    'glb_update_freq': 1000,
    'optim_epochs': 100,
}

ENV_CONFIG = {
    # 'algo': 'Multi-Agent',
    'num_agents': 8,
    'max_num_steps': 10000000,
    'device': 'cuda:0',
    'log_dir': '../../../../alta-logs/',
    'server_path' : CARLA_9_4_PATH,
    'server_binary' : CARLA_9_4_PATH + '/CarlaUE4.sh',
    'server_process' : None,
    # X Rendering Resolution
    'render_res_x' : 800,
    # Y Rendering Resolution
    'render_res_y' : 800,
    'sensor_x_res' : '80',
    'sensor_y_res' : '160',
    # Input X Res (Default set to Atari)
    'x_res': 84,
    # Input Y Res (Default set to Atari)
    'y_res': 84,
    'server_fps' : 10,
    'server_port' : None,
    'server_retries' : 5,
    'city_name' : 'Town01',
    'frame_skip': 1,
    'enable_planner' : True,
    'reward_function' : 'simple2',
    # Print measurements to screen
    'client' : None,
    ### 'discrete_actions': True,
    # Number of frames stacked together
    'framestack' : 1,
    ### 'grayscale' : False,
    'num_pedestrians' : 0,
    'max_steps' : 100000,
    'next_command': None,
    'verbose': False,
    'vehicle_type': 'vehicle.toyota.prius',
    'disable_two_wheeler' : True,
    'vehicle_types': ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola',
                      'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
                      'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
                      'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
                      'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
                      'vehicle.jeep.wrangler_rubicon'],
    'target_speed': 20,
    'sensors': ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation'],
    # 'action_type': 'discrete',
    'action_type': 'merged_speed',
    'sensor_tick': '0.0',
    'dist_for_success' : 10.0,
    'max_offlane_steps' : 100,
    'max_static_steps' : 100,
    'log_measurements_to_file': False,
    'sync_mode': True,
    # NOTE: crop does not work with framestack yet. need to add.
    'preprocess_crop_image': False,
    'scenarios' : 'navigation',
    'semantic' : False,
    'client_timeout_seconds' : 600,
    'enable_lane_invasion_sensor' : True,
    # 'carla_gpu': '0',
    'render_server': False,
    'steer_penalty_coeff': 0,
    'vae_encoding_norm_factor' : 10,
    'input_type': 'wp_angles_vecs_obs_info_speed_steer_ldist_light',
    'use_scenarios': True,
    'num_npc' : 20,
    'sample_npc': False,
    'num_npc_lower_threshold' : 70,
    'num_npc_upper_threshold' : 150,
    'binarized_image': False,
    'single_channel_image': False,
    'noise_dim' : 1,
    'const_collision_penalty': 250,
    'collision_penalty_speed_coeff': 250,
    'const_light_penalty': 250,
    'light_penalty_speed_coeff': 250,
    'terminate_on_light' : True,
    'enable_brake': True,
    # 'log_freq': 1,
    'zero_speed_threshold': 0.05,
    ### 'videos' : False,
    'obstacle_dist_norm' : 60,
    'spawn_points_fixed_idx' : [
        54, 234, 108,  12, 175,  71, 116,  99, 196,  63, 205,  46,  96,
       246, 128, 106, 143,  39,  72, 176, 140, 138,  91,  88, 241,  29,
        28, 238, 119, 221, 163,  81,  47, 255, 235,  64, 216, 151, 145,
        77,  35,  56,  68,  49, 154, 149, 201,  27, 212, 195, 230, 157,
         3,   5,  20, 193,   6,  90,  18,  13, 139,  44, 122, 220, 125,
       115,  43,   4, 213,  30,  62, 242, 219, 171,  41, 203,  57, 248,
       204, 226, 245, 135, 164, 153,  14, 188,   7, 123, 117, 222, 183,
       152, 150, 185, 224,  19, 104, 111,  82,  79,   0,  33,  38, 146,
        10, 173, 239,  32, 228, 209, 243, 200, 215, 236,  34,  84,  51,
        73,  53, 170, 217, 237, 102, 156,  45, 253,  37, 210, 118,  86,
        74,  61, 165, 179, 202, 101,  36, 132, 168, 137, 126, 178,  24,
         1, 247, 107,  93, 148,  50,  98,  87, 133, 162,   2, 214, 124,
       112, 211,  75, 121, 191, 113, 141,  26, 231, 174,  76, 207, 109,
       244, 129, 103,  52,  42,  55, 180,  89, 181,  69,  48,  21,  16,
       198,  66,  70, 130, 114,  15, 134,  40, 227, 223,  67,  78, 159,
       252, 147,  17, 166,  11, 131, 161, 105, 167,  95, 172, 233, 251,
       194,  60,  80, 182,  97,  59, 197,  25, 186, 136, 160, 120, 158,
       189, 192, 190, 187, 142, 232,   9, 127, 206, 169,  23, 208,  94,
       218,  83, 155,  65, 254, 249,  92, 240,  85, 100,  58,  22,   8,
       225,  31, 229, 250, 110, 177, 199, 184, 144],
    'test_fixed_spawn_points' : False,
    'train_fixed_spawn_points': False,
    'testing' : False,
    'disable_collision' : False,
    'enable_static' : True,
    'enable_obstacle_sensor': True,
    'use_pid_in_frame_skip' : True,
    'enable_lane_invasion_collision' : True,
    'vehicle_proximity_threshold' : 15,
    'traffic_light_proximity_threshold' : 10,
    'min_dist_from_red_light' : 4,
    'clip_reward' : False,
    'default_obs_traffic_val': 1,
    'reward_normalize_factor': 1,
    'success_reward': 0,
    'constant_positive_reward': 0,
    'frame_stack_size' : 1,
    'num_episodes' : 1,
    'disable_traffic_light': False,
    'disable_obstacle_info' : False,
    'test_comparison': False,
    'test_with_automatic_control': False,
    'updated_scenarios': False,
    'use_route_to_plan' : False,
    'discrete_actions': DISCRETE_ACTIONS,
    'episode_measurements': EPISODE_MEASUREMENTS,
}

ENV_CONFIG.update(SAC_CONFIG)
ENV_CONFIG.update(PPO_CONFIG)
