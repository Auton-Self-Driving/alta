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
    'control_steer': 0,
}

A2C_CONFIG = {
    'checkpoint': None,
    'policy_lr': 1e-3,
    'glb_update_freq': 5,
}

SAC_CONFIG = {
    'save_suffix': 'Expl10KTrain10KTarget1',
    'checkpoint': '',
    # 'checkpoint': './ckptSACx6_3300000_Feb010608AM57.pth',
    # 'checkpoint': './ckptSACx8_3300000_Feb010609AM48.pth',
    # 'checkpoint': './ckptSACx8_pidfixed_200000_Feb040617PM37.pth',
    # 'checkpoint': './ckptSACx1_Expl10kTrain10kTarget1_1200000_Aug291042AM31.pth',
    'policy_lr': 4e-4,
    'q_lr': 4e-4,
    'alpha_lr': 4e-5,
    'buffer_len': 1000000,
    'target_entropy': -1.,
    'tau': .01,
    'batch_size': 512,
    'q_update_freq': 25,
    'target_update_freq': 1,
    'explore_before': 10000,
    'train_after': 10000,
}

DSAC_CONFIG = {
    'save_suffix': 'Explrand10kTrain10k1Q1B25Efixlog0SteerScale0d5NoGoal',
    'checkpoint': '',
    'policy_lr': 4e-4,
    'q_lr': 4e-4,
    'alpha_lr': 4e-5,
    'num_workers': 8,
    'num_servers': 1, # currently only support 1 server
    'num_threads_per_server': 1,
    # 'device_list': ['cuda:1'],
    # 'device_list': ['cuda:0', 'cuda:1'],
    # 'device_list': ['cuda:2', 'cuda:3'],
    # 'device_list': ['cuda:0', 'cuda:1', 'cuda:2'],
    'device_list': ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    # 'device_list': ['cuda:2'],
    'buffer_len': 1000000,
    'log_alpha': 0,
    'log_target_entropy': -2, # not used if ent_autotune is False
    'ent_autotune': False,
    'tau': .01,
    'batch_size': 512,
    'q_update_freq': 1,
    'buffer_update_freq': 25,
    'target_update_freq': 1,
    'explore_before': 10000,
    'explore_mode': 'random',
    # 'explore_mode': 'autopilot',
    # 'explore_mode': 'transfuser_autopilot',
    'train_after': 10000,
    'standard': True,
    # 'standard': False, # to collect traj
    'push_grad': False,
}

DPPO_CONFIG = {
    'save_suffix': '24dim_10wp_nocrach_dense_no_lane_term_tanh_squashed', # 
    'checkpoint': 'ckptDPPO1x14x8_24dim_10wp_nocrach_dense_no_lane_term_tanh_squashed_17739845_Jan260447PM53.pth',# 'ckptDPPO1x14x8_15dim_nocrach_dense_no_lane_term_tanh_squashed_sp_30_wp_10_5710454_Jan080933PM30.pth', # 'ckptDPPO1x14x8_7dim_nocrach_dense_no_lane_3002333_Dec130112AM03.pth',#'ckptDPPO1x14x8_15dim_nocrach_dense_no_lane_term_tanh_squashed_gamma_08_7508753_Dec060659PM49.pth',#ckptDPPO1x15x8_test2_nolane_15dim_nocrach_empty_8379736_Nov021006AM17.pth',
    # 'ckpt_mode': 'load',
    'ckpt_mode': '',
    # 'ckpt_mode': 'resume',
    'gamma': 0.99,
    'policy_lr': 4e-4,
    'eps_clip': .2,
    'grad_clip': .5,
    'squash': True, # This enables tanh squashing of sampled actions from policy
    'focal_loss': False,
    'standard': False, # if False, will push traj after finishing an episode
    'push_grad': False,
    'num_workers': 14,#14, #14,#1,#
    'num_servers': 1, # currently only support 1 server
    'num_threads_per_server': 1,
    # 'device_list': ['cuda:2'],
    'device_list': ['cuda:1', 'cuda:0', 'cuda:3'],
    # 'device_list': ['cuda:0','cuda:1', 'cuda:2', 'cuda:3'],
    'worker_grad_update_freq': 20000,
    'worker_optim_epochs': 10,
    'server_glb_update_freq': 100,
    'server_adaptive_freq': True,
    'save_freq': 300000,
}

PPO_CONFIG = {
    'save_suffix': 'O10mL10mG0m1kupdstdoffchallenge',
    'checkpoint': './checkpoints/15dim_nocrach_dense_no_lane_term_tanh_squashed/ckptDPPO1x14x8_15dim_nocrach_dense_no_lane_term_tanh_squashed_2702504_Nov281208AM12.pth',
    'policy_lr': 4e-4,
    'eps_clip': .2,
    'grad_clip': .5,
    'squash': True, # This enables tanh squashing of sampled actions from policy
    'nesterov': False,
    'standard': False,
    # 'focal_loss': [.5, .5],
    'focal_loss': False,
    'glb_update_freq': 1000,
    'optim_epochs': 10,
}

OFFLINE_CONFIG = {
    'epoch': 40,
    'offline_repo_location': '/zfsauton2/home/zhehuang/Documents/transformer_rl',
    'dvae_iql_policy_location': '/zfsauton2/home/zhehuang/Documents/transformer_rl/logs/iql/lowerdimobs-random-ttc/newexp_iql_seed2/',
    'offline_policy_location': '/zfsauton2/home/zhehuang/Documents/transformer_rl/logs/dvae_dt/lowerdimobs-random-ttc/dvae_bt_newset8_seed6/',
}


TEST_CONFIG = {
    'PPO': True, # else SAC, currently only support those two
    'checkpoint': './checkpoints/15dim_nocrach_dense_no_lane_term_tanh_squashed/ckptDPPO1x14x8_15dim_nocrach_dense_no_lane_term_tanh_squashed_7216852_Nov290726AM24.pth',
    'num_agents': 1,
    'num_npc': 70,
    'sample_npc': False,
    'scenarios' : 'no_crash_dense', # 'no_crash_empty', # 
    # 'scenarios' : 'challenge_test_scenario',
    'use_scenarios': True,
    'city_name' : 'Town02', # Set to town 2
    'num_episodes' : 25,
    'target_speed': 50, # DEFAULT TO 50,
    'steering_scale': 0.5,
    'testing' : False, # spawn point pending bugs in env line#142
    'enable_static_termination' : True,
    'enable_obstacle_sensor': True,
    'obs_cosine_velocity': True,
    'check_obs_same_lane': True,
    'obs_sensor_vehicle_only': True,
    'front_obs_sensor_hit_radius': .5,
    'side_obs_sensor_hit_radius': .7854,
    'disable_traffic_light': False,
    'terminate_on_light' : False,
    'enable_lane_invasion_termination' : True,
    'front_obs_proximity_threshold' : 15,
    'side_obs_proximity_threshold' : 5,
    'traffic_light_proximity_threshold' : 15,
    'min_dist_from_red_light' : 0,
    'npc_reset_freq': None,
    'verbose': False,
    'weak_verbose': False,
    'test_verbose': True,
    # 'test_verbose': False,
    'sensor_x_res' : '400',
    'sensor_y_res' : '800',
    'videos': True,
    'save_buffer': False,
}

ENV_CONFIG = {
    'algo': 'Multi-Agent',
    'num_envs': 1,
    'num_agents': 8,#1,#
    'max_num_steps': 25000000,
    'num_episodes' : 1,
    'max_steps' : 50000,
    'next_command': None,  
    
    'initial_town' : 'Town01', # no ldb training routes for town05
    'avail_town_list': ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07'],
    
    # NOTE: crop does not work with framestack yet. need to add.
    'preprocess_crop_image': False,
    # Refer to _set_scenario in carla_env for scenarios list
    'scenarios' : 'no_crash_dense', # 'no_crash_empty', 'straight', 'challenge_train_scenario', 'navigation', 'leaderboard_navigation', # Use this for ldb
    'updated_scenarios': False,
    'use_scenarios': True,

    'client' : None,
    'client_timeout_seconds' : 6000,




    ############### AGENT HYPER PARAMS ###############

    # 'input_type': 'wp_obs_info_speed_steer_ldist_light', # 7-dim
    # 'input_type': 'wp_obs_more_info_steer_ldist_light', # 14-dim 
    # 'input_type': 'wp_obs_more_info_speed_steer_ldist_light', # 15-dim 
    # 'input_type': 'wp_2avg_obs_more_info_speed_steer_ldist_light', # 16-dim 
    'input_type': 'wp_list_obs_more_info_steer_ldist_light', # >=14-dim 
    # 'input_type': 'wp_list_obs_more_info_speed_steer_ldist_light', # >=15-dim 
    'action_type': 'merged_speed_scaled_tanh',
    'vae_encoding_norm_factor' : 10,
    'enable_brake': True,
    'target_speed': 50, #50, ##### REDUCED FOR AN ABLATION | DEFAULT = 50
    'steering_scale': 0.5,
    'num_waypoints': 5, #10, ##### DEFAULT = 5
    'frame_skip': 1,
    'noise_dim' : 1,
    'use_pid_in_frame_skip' : True,
    'discrete_actions': DISCRETE_ACTIONS,
    'episode_measurements': EPISODE_MEASUREMENTS,
    'enable_planner' : True,
    'use_route_to_plan' : False,
    'frame_stack_size' : 1,
    'framestack' : 1, # Number of frames stacked together




    ############### ENVIRONMENT HYPERPARAMETERS ###############

    # Server Settings

    'device': 'cuda:2', # Redundant. Its over written by the device list of the agent
    'log_dir': '../../../../alta-logs/',
    'server_path' : CARLA_9_4_PATH,
    'server_binary' : CARLA_9_4_PATH + '/CarlaUE4.sh',
    'server_process' : None,
    # X Rendering Resolution
    'render_res_x' : 800,
    # Y Rendering Resolution
    'render_res_y' : 800,
    'sensor_x_res' : '1',
    'sensor_y_res' : '1',
    # Input X Res (Default set to Atari)
    'x_res': 84,
    # Input Y Res (Default set to Atari)
    'y_res': 84,
    'server_fps' : 10,
    'server_port' : None,
    'server_retries' : 5,
    'sync_mode': True,
    'render_server': False,
    'binarized_image': False,
    'single_channel_image': False,
    'verbose': False,
    'weak_verbose': False,
    
    # NPC Vehicle Settings
    'num_npc' : 0,
    'sample_npc': True,
    'num_npc_lower_threshold' : 20,
    'num_npc_upper_threshold' : 380,
    'npc_reset_freq': 10000, # CHECK: Basically means never reset NPC?

    # Actor Spawning
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
    'test_fixed_spawn_points': False,
    'train_fixed_spawn_points': False,

    # ENV Asset Settings
    'disable_traffic_light': False,
    'disable_obstacle_info' : False,
    'min_num_eps_before_switch_town': 15,
    'vehicle_type': 'vehicle.toyota.prius',
    'disable_two_wheeler' : True,
    'vehicle_types': ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola',
                      'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
                      'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
                      'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
                      'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
                      'vehicle.jeep.wrangler_rubicon'],





    ############### MEAUSUREMENT HYPERPARAMETERS ###############

    'log_measurements_to_file': False,
    # 'log_freq': 1,

    'dist_for_success' : 10.0,
    'max_offlane_steps' : 0,
    'max_static_steps' : 200,
    'zero_speed_threshold': 0.05,
    'obstacle_dist_norm' : 60,
    'num_pedestrians' : 0,

    # Sensor Settings
    'sensors': ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation'],
    'semantic' : False,
    # 'grayscale' : False,
    'sensor_tick': '0.0',
    'enable_obstacle_sensor': True, 
    'obs_sensor_vehicle_only': False,
    'obs_cosine_velocity': True,
    'check_obs_same_lane': True,
    'front_obs_sensor_hit_radius': .5,
    'side_obs_sensor_hit_radius': .7854, # pi / 4
    'all_obs_hit_radius': 2,
    'enable_lane_invasion_sensor' : False, # DISABLED [AMAN]    
    'front_obs_proximity_threshold' : 30,
    'side_obs_proximity_threshold' : 5,
    'all_obs_proximity_threshold' : 45,
    'vehicle_proximity_threshold' : 45,
    'traffic_light_proximity_threshold' : 20,
    'min_dist_from_red_light' : 0,
    'default_obs_traffic_val': 1,






    ############### REWARD HYPERPARAMETERS ###############

    'reward_function' : 'simple2', # Options - 'obs', 'simple2_modified', 'simple3'

    # Reward Coefficients
    'steer_penalty_coeff': 0,
    'const_collision_penalty': 250,
    'collision_penalty_speed_coeff': 250,
    'const_light_penalty': 250,
    'light_penalty_speed_coeff': 250,
    'static_penalty': 0,
    'reward_normalize_factor': 1,
    'success_reward': 0,
    'constant_positive_reward': 0,

    # Reward and Termination Booleans
    'terminate_on_light' : False,
    'disable_collision': False,
    'enable_static_termination': False, # DISABLED [AMAN]
    'enable_lane_invasion_termination' : False, #DISABLED [AMAN]
    'enable_lane_invasion_collision' : False, #DISABLED [AMAN]
    'clip_reward' : False,
    'reward_verbose' : False, # TODO toggle






    ############### TEST SETTINGS ###############
    
    'testing': False,
    'test_verbose': False,
    'test_comparison': False,
    'test_with_automatic_control': False,
    
    
    
}
