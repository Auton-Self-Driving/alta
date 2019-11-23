import os
import sys
import glob

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+'/**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(".egg file not found! Kindly check for your Carla installation.")
    pass

DEFAULT_ENV = {
    "server_path" : CARLA_9_4_PATH,
    "server_binary" : CARLA_9_4_PATH + '/CarlaUE4.sh',
    "server_process" : None,
    # X Rendering Resolution
    "render_res_x" : 800,
    # Y Rendering Resolution
    "render_res_y" : 800,
    "sensor_x_res" : '128',
    "sensor_y_res" : '128',
    # Input X Res (Default set to Atari)
    "x_res": 84,
    # Input Y Res (Default set to Atari)
    "y_res": 84,
    "server_fps" : 10,
    "server_port" : None,
    "server_retries" : 5, 
    "city_name" : "Town01",
    "frame_skip": 1,
    "enable_planner" : True,
    "reward_function" : 'corl',
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions": False,

    # Number of frames stacked together
    "framestack" : 1,
    "grayscale" : False,
    "num_pedestrians" : 0,
    "max_steps" : 10000,
    "next_command": None,
    "verbose": False,
    "vehicle_type": 'vehicle.toyota.prius',
    "disable_two_wheeler" : False,
    "vehicle_types": ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola', 
                      'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
                      'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
                      'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
                      'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
                      'vehicle.jeep.wrangler_rubicon'],
    "target_speed": 20,
    "sensors": ["sensor.camera.rgb", "sensor.camera.semantic_segmentation"],
    "action_type": "merged_gas",
    "sensor_tick": '0.0',
    "dist_for_success" : 10.0,
    "max_offlane_steps" : 20,
    "max_static_steps" : 200,
    "log_measurements_to_file": False,
    "train_config": None,
    "sync_mode": True,
    # NOTE: crop does not work with framestack yet. need to add.
    "preprocess_crop_image": False,
    "scenarios" : "straight",
    "semantic" : False,
    "client_timeout_seconds" : 600,
    "enable_lane_invasion_sensor" : True,
    "carla_gpu": "0",
    "render_server": False,
    "steer_penalty_coeff": 0,
    "vae_encoding_norm_factor" : 10,
    "input_type": None,
    "use_scenarios": True,
    "num_npc" : 0,
    "train_vae" : False,
    "noise_dim" : 1,
    "const_collision_penalty": 0,
    "collision_penalty_speed_coeff": 0,
    "enable_brake": False,
    "log_freq": 1,
    "zero_speed_threshold": 0.05, 
    "videos" : False,
    "obstacle_dist_norm" : 60
}

episode_measurements = {
    "episode_id": None,
    "num_steps": None,
    "location": None,
    "speed": None,
    "distance_to_goal": None,
    "num_collisions": 0,
    "num_laneintersections": 0,
    "static_steps": 0,
    "offlane_steps": 0
    # intersection_offroad
    # intersection_otherlane
    # next_command
}

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.0, 0.0],
#     # Forward
#     1: [0.5, 0.0],
#     # Forward left
#     2: [0.25, -0.3],
#     3: [0.25, -0.1],
#     # Forward right
#     4: [0.25, 0.1],
#     5: [0.25, 0.3],
#     # Brake
#     6: [-0.5, 0.0],
#     # Brake left
#     7: [-0.25, -0.3],
#     8: [-0.25, -0.1],
#     # Brake right
#     9: [-0.25, 0.1],
#     10: [-0.25, 0.3]
# }

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [10.0, 0.0],
#     # Forward
#     1: [20.0, 0.0],
#     # Forward left
#     2: [15.0, -0.3],
#     3: [15.0, -0.1],
#     # Forward right
#     4: [15.0, 0.1],
#     5: [15.0, 0.3],
#     # Brake
#     6: [0.0, 0.0],
#     # Brake left
#     7: [5.0, -0.3],
#     8: [5.0, -0.1],
#     # Brake right
#     9: [5.0, 0.1],
#     10: [5.0, 0.3]
# }

DISCRETE_ACTIONS = {
    # Coast
    0: [10.0, -0.5],
    # Forward
    1: [10.0, -0.4],
    # Brake
    2: [10.0, -0.3],
    # Left
    3: [10.0, -0.2],
    # Right
    4: [10.0, -0.1],
    # Forward left
    5: [10.0, 0.0],
    # Forward right
    6: [10.0, 0.1],
    # Brake left
    7: [10.0, 0.2],
    # Brake right
    8: [10.0, 0.3],

    9: [10.0, 0.4],
    10: [10.0, 0.5]
}

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.0, 0.0],
#     # Forward
#     1: [2.0, 0.0],
#     # Forward left
#     2: [4.0, 0.0],
#     3: [6.0, 0.0],
#     # Forward right
#     4: [8.0, 0.0],
#     5: [10.0, 0.0],
#     # Brake
#     6: [12.0, 0.0],
#     # Brake left
#     7: [14.0, 0.0],
#     8: [16.0, 0.0],
#     # Brake right
#     9: [18.0, 0.0],
#     10: [20.0, 0.0]
# }


class ConfigManager(object):
    def __init__(self, algo='DDPG'):
        self.config = {}

        self._initialize_config(algo)

    def _initialize_config(self, algo):
        if algo == 'DDPG':
            self.config["algo"] = "DDPG"
            self.config["x_res"] = 200
            self.config["y_res"] = 84
            self.config["reward_function"] = "cirl"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "torch"
            self.config["action_type"] = "merged_gas"
        elif algo == 'DQN':
            self.config["algo"] = "DQN"
            self.config["x_res"] = 84
            self.config["y_res"] = 84
            self.config["reward_function"] = "simple"
            self.config["discrete_actions"] = True
            self.config["train_config"] = "baselines"
            self.config["action_type"] = "sep_gas"
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["scenarios"] = "straight"
        elif algo == 'PPO':
            self.config["algo"] = "PPO"
            self.config["reward_function"] = "simple2"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "merged_speed_scaled_tanh"
            self.config["preprocess_crop_image"] = True
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["semantic"] = True
            self.config["scenarios"] = "navigation"
            self.config["videos"] = False
            self.config["x_res"] = 80
            self.config["y_res"] = 160
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = False
            self.config["carla_gpu"] = "1"
            self.config["disable_two_wheeler"] = True
            self.config["enable_lane_invasion_sensor"] = True
        elif algo == 'SAC':
            self.config["algo"] = "SAC"
            self.config["reward_function"] = "simple2"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "merged_speed"
            self.config["preprocess_crop_image"] = True
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["semantic"] = True
            self.config["scenarios"] = "straight"
            self.config["videos"] = True
            self.config["x_res"] = 80
            self.config["y_res"] = 160
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = False
            self.config["carla_gpu"] = "1"
        elif algo == 'AE':
            self.config["algo"] = "AE"
            self.config["action_type"] = "control"
            self.config["use_scenarios"] = False
            self.config["semantic"] = True
            self.config['max_steps'] = 10000
            self.config["city_name"] = "Town01"
            self.config["num_npc"] = 60
            self.config["input_type"] = "ae_train"
            self.config["videos"] = True
        elif algo == 'PID_TUNE':
            self.config["algo"] = "PPO"
            self.config["reward_function"] = "simple2"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "merged_speed_pid_test"
            self.config["preprocess_crop_image"] = True
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["semantic"] = False
            self.config["scenarios"] = "straight_dynamic"
            self.config["videos"] = False
            self.config["x_res"] = 80
            self.config["y_res"] = 160
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = True
            self.config["carla_gpu"] = "1"
            self.config["max_static_steps"] = 20
