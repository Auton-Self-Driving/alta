from gym.spaces import Box, Discrete
import numpy as np

class ObservationsManager:

    def __init__(self, config, world):

        self.config = config
        self._world = world
        self.action_space = None
        self.observation_space = None

    def setup_observation_and_action_space(self):

            # TODO: Verify the limits and bounds of observation spaces
            if self.config["action_type"] == 'merged_gas':
                # Streer, Throttle
                self.action_space = Box(low=np.array([-0.5, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float32)
            elif self.config["action_type"] == 'merged_speed':
                # Steer, Speed
                self.action_space = Box(low=np.array([-0.5, -10.0]), high=np.array([0.5, 10.0]), dtype=np.float32)
            elif self.config["action_type"] == 'merged_speed_tanh' or self.config["action_type"] == 'merged_speed_scaled_tanh':
                # Steer, Speed
                self.action_space = Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)
            elif self.config["action_type"] == "merged_speed_pid_test":
                self.action_space = Box(low=np.array([-0.5, -20.0]), high=np.array([0.5, 20.0]), dtype=np.float32)
            elif self.config["action_type"] == 'steer_only':
                # Steer only
                self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
            elif self.config["action_type"] == 'discrete':
                # Discrete actions
                self.action_space = Discrete(len(self.config['discrete_actions']))
            elif self.config["action_type"] == 'control':
                # Discrete actions
                self.action_space = Discrete(len(self.config['discrete_actions']))
            elif self.config["action_type"] == "cubic_bezier_3dof":
                self.action_space = Box(low=np.array([0.0, -6.0,-4.0,-1.0]), 
                                        high=np.array([4.0, 6.0, 4.0,1.0]), dtype=np.float32)
            elif self.config["action_type"] == "cubic_bezier_5dof":
                self.action_space = Box(low=np.array([1.0, -4.0, 0.0,-4.0,-4.0,-1.0]), 
                                        high=np.array([4.0, 4.0, 4.0,4.0,4.0,1.0]), dtype=np.float32)
            elif self.config["action_type"] == "cubic_bezier_5dof_disc_thrt":
                self.action_space = Box(low=np.array([1.0, -4.0, 0.0,-4.0,-4.0,-1.0]), 
                                        high=np.array([4.0, 4.0, 4.0,4.0,4.0,1.0]), dtype=np.float32)
            elif self.config["action_type"] == "speed_wp":
                self.action_space = Box(low=np.array([0,-6.0,-1.0]), 
                                        high=np.array([6,6.0,1.0]), dtype=np.float32)

            if self.config["input_type"] == 'wp':
                self.observation_space = Box(low=np.array([-4.0]), high=np.array([4.0]), dtype=np.float32)
            elif self.config["input_type"] in ['wp_constant', 'wp_noise', 'wp_obs_dist', 'wp_obs_bool']:
                self.observation_space = Box(low=np.array([[-4.0, -1.0]]), high=np.array([[4.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_ldist_goal':
                self.observation_space = Box(low=np.array([[-4.0, -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_bool_noise':
                limit = np.hstack((np.array([[4]]), np.ones((1, 1 + self.config["noise_dim"]))))
                self.observation_space = Box(low=-limit, high=limit, shape=(1, 2 + self.config["noise_dim"]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_speed':
                self.observation_space = Box(low=np.array([[-4.0, 0.0]]), high=np.array([[4.0, 12.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_speed_goal':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_speed_steer_goal':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, 0.0]]), high=np.array([[4.0, 1.0, 0.5, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_speed_steer_goal_obs_bool':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 0.5, 10.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, -0.5, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -1 * self.config['steering_scale'], -1.0, 0.0, 0.0]]),
                    high=np.array([[4.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_light': # currently using
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, -1., 0., -1., 0., 0.0, -self.config['steering_scale'], -1.0, 0.0]]),
                high=np.array([[4.0, 1.0, 1.0, 1., 1., 1., 1., 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_more_info_steer_ldist_light': # 14 dim obs space
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0]]),
                high=np.array([[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_more_info_speed_steer_ldist_light': # 15 dim obs space w/ 5 obs sensors
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0]]),
                high=np.array([[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_2avg_obs_more_info_speed_steer_ldist_light': # 16 dim obs space w/ 5 obs sensors
                self.observation_space = Box(low=np.array([[-4.0,-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0]]),
                high=np.array([[4.0,4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_list_obs_more_info_steer_ldist_light': # >=14 dim obs space w/ 5 obs sensors and no speed measure
                lower_bound = [-4.0] * self.config['num_waypoints']
                lower_bound.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0])
                upper_bound = [4.0] * self.config['num_waypoints']
                upper_bound.extend([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0])
                self.observation_space = Box(low=np.array([lower_bound]),
                high=np.array([upper_bound]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_list_obs_more_info_speed_steer_ldist_light': # >=15 dim obs space w/ 5 obs sensors
                lower_bound = [-4.0] * self.config['num_waypoints']
                lower_bound.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0])
                upper_bound = [4.0] * self.config['num_waypoints']
                upper_bound.extend([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0])
                self.observation_space = Box(low=np.array([lower_bound]),
                high=np.array([upper_bound]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_angles_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-4.0, -4.0, -4.0, -4.0, -4.0, -1.0, -1.0, 0.0, -0.5, -1.0, -1.0]]),
                                                high=np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vecs_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -0.5, -1.0, -1.0]]),
                                        high=np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_angles_vecs_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -0.5, -1.0, -1.0]]),
                                        high=np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'vae':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 400), dtype=np.float32)
            elif self.config["input_type"] == 'transformer':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(800,), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 401), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 404), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        # shape=(1, 406), dtype=np.float32) # Model used for Learning to drive using Waypoints (last layer dim = 16)
                                        shape=(1, 1606), dtype=np.float32) # Model used for Learning to Drive with Dynamic Actors (last layer dim = 64)
            elif self.config["input_type"] == 'wp_360_obstacle_speed_steer':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
                                high=np.array([[4.0, 1.0, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]),
                                dtype=np.float32)
            elif self.config["input_type"] == 'wp_360_obstacle_steer':
                self.observation_space = Box(low=np.array([[-4.0, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
                                high=np.array([[4.0, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]),
                                dtype=np.float32)
            elif self.config["input_type"] == 'wp_360_obstacle':
                self.observation_space = Box(low=np.array([[-4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
                                high=np.array([[4.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]),
                                dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae_obs_info_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        # shape=(1, 408), dtype=np.float32) # Model used for Learning to drive using Waypoints (last layer dim = 16)
                                        shape=(1, 1608), dtype=np.float32) # Model used for Learning to Drive with Dynamic Actors (last layer dim = 64)steer_ldist_goal_light':
            elif self.config["input_type"] == 'wp_cnn_obs_info_speed_steer_ldist_goal_light' or self.config["input_type"] == 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light':
                if not self.config["single_channel_image"]:
                    if self.config["binarized_image"]:
                        dim = 2
                    else:
                        dim = 5
                else:
                    dim = 1
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, (int(self.config['sensor_y_res']) * int(self.config['sensor_x_res']) * dim * self.config['frame_stack_size']) + 8), dtype=np.float32)

    def create_observations(self, agent, obs): # Observation created here.
            obs['observation'] = np.array([agent.episode_measurements['next_orientation']])

            if self.config["input_type"] == 'wp_constant':
                obs['observation'] = np.array([agent.episode_measurements['next_orientation'], 0.0])

            elif self.config["input_type"] == 'wp_noise':
                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))

            elif self.config["input_type"] == 'wp_obs_dist':
                obs_dist = agent.episode_measurements['obstacle_dist'] / self.config["obstacle_dist_norm"]
                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_dist])))

            elif self.config["input_type"] == 'wp_obs_bool':
                obs_bool = agent.episode_measurements['obstacle_visible']
                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_bool])))

            elif self.config["input_type"] == 'wp_ldist_goal':
                ldist = agent.episode_measurements['dist_to_trajectory']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([ldist]), np.array([distance_to_goal_trajec])))

            elif self.config["input_type"] == 'wp_obs_bool_noise':
                obs_bool = agent.episode_measurements['obstacle_visible']
                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_bool]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))

            elif self.config["input_type"] == 'wp_speed':
                obs_speed = agent.episode_measurements['speed'] / 10
                obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed])))

            elif self.config["input_type"] == 'wp_speed_goal':
                obs_speed = agent.episode_measurements['speed'] / 10
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 100
                obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([distance_to_goal_trajec])))

            elif self.config["input_type"] == 'wp_speed_steer_goal':
                obs_speed = agent.episode_measurements['speed'] / 10
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 100
                steer = agent.episode_measurements['control_steer']
                obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([steer]), np.array([distance_to_goal_trajec])))

            elif self.config["input_type"] == 'wp_speed_steer_goal_obs_bool':
                obs_speed = agent.episode_measurements['speed'] / 10
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 100
                steer = agent.episode_measurements['control_steer']
                obs_bool = agent.episode_measurements['obstacle_visible']
                obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([steer]), np.array([distance_to_goal_trajec]), np.array([obs_bool])))

            elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':

                speed = agent.episode_measurements['speed'] / 10
                obs_bool = agent.episode_measurements['obstacle_visible']
                steer = agent.episode_measurements['control_steer']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
                light = agent.episode_measurements['red_light_dist']

                # normalization
                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_bool]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec]), np.array([light])))

            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':

                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
                light = agent.episode_measurements['red_light_dist']

                # normalization

                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal':

                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500

                # normalization

                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec])))

            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_light': # 7 dim

                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer'] #agent.trajectory_yaw_drift / 360. # agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                # normalization

                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

            elif self.config["input_type"] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                obstacle_dist_left = agent.episode_measurements['obstacle_dist_left']
                obstacle_speed_left = agent.episode_measurements['obstacle_speed_left']
                obstacle_dist_right = agent.episode_measurements['obstacle_dist_right']
                obstacle_speed_right = agent.episode_measurements['obstacle_speed_right']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                # normalization

                if obstacle_dist <= self.config['front_obs_proximity_threshold']:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_dist_left <= self.config['front_obs_proximity_threshold']:
                    obstacle_dist_left = obstacle_dist_left / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist_left = self.config['default_obs_traffic_val']

                if obstacle_dist_right <= self.config['front_obs_proximity_threshold']:
                    obstacle_dist_right = obstacle_dist_right / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist_right = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if obstacle_speed_left != -1:
                    obstacle_speed_left = obstacle_speed_left / 20
                else:
                    obstacle_speed_left = self.config['default_obs_traffic_val']

                if obstacle_speed_right != -1:
                    obstacle_speed_right = obstacle_speed_right / 20
                else:
                    obstacle_speed_right = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]),
                np.array([obstacle_dist_left]), np.array([obstacle_speed_left]), np.array([obstacle_dist_right]), np.array([obstacle_speed_right]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

            elif self.config["input_type"] == 'wp_obs_more_info_steer_ldist_light': # 14dim no speed obs space

                feat_list = [agent.episode_measurements['next_orientation']]

                for suffix, sensor in agent.obstacle_sensor.items():
                    obstacle_dist = agent.episode_measurements['obstacle_dist_{}'.format(suffix)]
                    obstacle_speed = agent.episode_measurements['obstacle_speed_{}'.format(suffix)]
                    # normalization
                    if obstacle_dist <= sensor.max_distance:
                        obstacle_dist = obstacle_dist / sensor.max_distance
                    else:
                        obstacle_dist = self.config['default_obs_traffic_val']

                    if obstacle_speed != -1:
                        obstacle_speed = obstacle_speed / 20
                    else:
                        obstacle_speed = self.config['default_obs_traffic_val']
                    feat_list.extend([obstacle_dist, obstacle_speed])

                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                feat_list.extend([steer, ldist, light])

                obs['observation'] = np.array(feat_list)

            elif self.config["input_type"] == 'wp_obs_more_info_speed_steer_ldist_light': # 15dim obs space

                feat_list = [agent.episode_measurements['next_orientation']]

                for suffix, sensor in agent.obstacle_sensor.items():
                    obstacle_dist = agent.episode_measurements['obstacle_dist_{}'.format(suffix)]
                    obstacle_speed = agent.episode_measurements['obstacle_speed_{}'.format(suffix)]
                    # normalization
                    if obstacle_dist <= sensor.max_distance:
                        obstacle_dist = obstacle_dist / sensor.max_distance
                    else:
                        obstacle_dist = self.config['default_obs_traffic_val']

                    if obstacle_speed != -1:
                        obstacle_speed = obstacle_speed / 20
                    else:
                        obstacle_speed = self.config['default_obs_traffic_val']
                    feat_list.extend([obstacle_dist, obstacle_speed])

                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                feat_list.extend([speed, steer, ldist, light])

                obs['observation'] = np.array(feat_list)

            elif self.config["input_type"] == 'wp_2avg_obs_more_info_speed_steer_ldist_light': # 16 dim obs space

                feat_list = [agl for agl in agent.next_wp_angles] # First entry is furthest Wp

                if len(feat_list) < 10: # If not enuf waypoints, replicate last waypoint (which will be destination)
                    for itr in range(10-len(feat_list)):
                        feat_list.append(feat_list[-1])

                first_avg, last_avg = sum(feat_list[0:5])/5.,sum(feat_list[-5:])/5.
                feat_list = [first_avg, last_avg]

                for suffix, sensor in agent.obstacle_sensor.items():
                    obstacle_dist = agent.episode_measurements['obstacle_dist_{}'.format(suffix)]
                    obstacle_speed = agent.episode_measurements['obstacle_speed_{}'.format(suffix)]
                    # normalization
                    if obstacle_dist <= sensor.max_distance:
                        obstacle_dist = obstacle_dist / sensor.max_distance
                    else:
                        obstacle_dist = self.config['default_obs_traffic_val']

                    if obstacle_speed != -1:
                        obstacle_speed = obstacle_speed / 20
                    else:
                        obstacle_speed = self.config['default_obs_traffic_val']
                    feat_list.extend([obstacle_dist, obstacle_speed])

                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                feat_list.extend([speed, steer, ldist, light])

                obs['observation'] = np.array(feat_list)

            elif self.config["input_type"] == 'wp_list_obs_more_info_steer_ldist_light': # Variable >= 14 dim obs space. No speed

                feat_list = [agl for agl in agent.next_wp_angles] # First entry is furthest Wp

                if len(feat_list) == 0: # Raise Error
                    print("[carla_env.create_observation] Next Wp List Length is 0!!!")
                    print(agent.episode_measurements['next_orientation'])
                    print(agent.next_waypoints)
                    print(agent.global_planner._waypoints_queue)

                elif len(feat_list) < 10: # If not enuf waypoints, replicate last waypoint (which will be destination)
                    for itr in range(10-len(feat_list)):
                        feat_list.append(feat_list[-1])

                for suffix, sensor in agent.obstacle_sensor.items():
                    obstacle_dist = agent.episode_measurements['obstacle_dist_{}'.format(suffix)]
                    obstacle_speed = agent.episode_measurements['obstacle_speed_{}'.format(suffix)]
                    # normalization
                    if obstacle_dist <= sensor.max_distance:
                        obstacle_dist = obstacle_dist / sensor.max_distance
                    else:
                        obstacle_dist = self.config['default_obs_traffic_val']

                    if obstacle_speed != -1:
                        obstacle_speed = obstacle_speed / 20
                    else:
                        obstacle_speed = self.config['default_obs_traffic_val']
                    feat_list.extend([obstacle_dist, obstacle_speed])

                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                feat_list.extend([steer, ldist, light])

                obs['observation'] = np.array(feat_list)

            elif self.config["input_type"] == 'wp_list_obs_more_info_speed_steer_ldist_light': # Variable >= 15 dim obs space

                feat_list = [agl for agl in agent.next_wp_angles] # First entry is furthest Wp

                if len(feat_list) == 0: # Raise Error
                    print("[carla_env.create_observation] Next Wp List Length is 0!!!")
                    print(agent.episode_measurements['next_orientation'])
                    print(agent.next_waypoints)
                    print(agent.global_planner._waypoints_queue)

                elif len(feat_list) < 10: # If not enuf waypoints, replicate last waypoint (which will be destination)
                    for itr in range(10-len(feat_list)):
                        feat_list.append(feat_list[-1])

                for suffix, sensor in agent.obstacle_sensor.items():
                    obstacle_dist = agent.episode_measurements['obstacle_dist_{}'.format(suffix)]
                    obstacle_speed = agent.episode_measurements['obstacle_speed_{}'.format(suffix)]
                    # normalization
                    if obstacle_dist <= sensor.max_distance:
                        obstacle_dist = obstacle_dist / sensor.max_distance
                    else:
                        obstacle_dist = self.config['default_obs_traffic_val']

                    if obstacle_speed != -1:
                        obstacle_speed = obstacle_speed / 20
                    else:
                        obstacle_speed = self.config['default_obs_traffic_val']
                    feat_list.extend([obstacle_dist, obstacle_speed])

                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                feat_list.extend([speed, steer, ldist, light])

                obs['observation'] = np.array(feat_list)

            elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec])))

            elif self.config["input_type"] == 'wp_vae_speed_steer_ldist_goal_light':
                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
                light = agent.episode_measurements['red_light_dist']

                # normalization
                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

            elif self.config["input_type"] in ['wp_vae_obs_info_speed_steer_ldist_goal_light', 'wp_cnn_obs_info_speed_steer_ldist_goal_light', 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:
                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
                light = agent.episode_measurements['red_light_dist']

                # normalization

                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

            elif self.config["input_type"] == 'wp_angles_obs_info_speed_steer_ldist_light':
                wp_angles_array, wp_vectors_array = env_util.get_wp_obs_input(agent)
                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.dist_to_trajectory
                light = agent.episode_measurements['red_light_dist']

                # normalization
                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']
                obs['observation'] = np.concatenate((wp_angles_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

            elif self.config["input_type"] == 'wp_vecs_obs_info_speed_steer_ldist_light':
                wp_angles_array, wp_vectors_array = env_util.get_wp_obs_input(agent)

                # normalize vectors by 10, assuming max norm of vector would be 10
                wp_vectors_array = wp_vectors_array / 10
                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.dist_to_trajectory
                light = agent.episode_measurements['red_light_dist']
                # normalization
                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']
                obs['observation'] = np.concatenate((wp_vectors_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

            elif self.config["input_type"] == 'wp_angles_vecs_obs_info_speed_steer_ldist_light':
                wp_angles_array, wp_vectors_array = env_util.get_wp_obs_input(agent)

                # normalize vectors by 10, assuming max norm of vector would be 10
                wp_vectors_array = wp_vectors_array / 10
                speed = agent.episode_measurements['speed'] / 10
                obstacle_dist = agent.episode_measurements['obstacle_dist']
                obstacle_speed = agent.episode_measurements['obstacle_speed']
                steer = agent.episode_measurements['control_steer']
                ldist = agent.dist_to_trajectory
                light = agent.episode_measurements['red_light_dist']
                # normalization
                if obstacle_dist != -1:
                    obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 10
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), wp_angles_array, wp_vectors_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

            elif self.config['input_type'] == 'transformer':
                sym_dict = fetch_symbolic_dict(agent)
                obs['observation'] = flatten_obs(sym_dict) # (1, 100, 8)

            elif self.config['input_type'] == 'wp_360_obstacle_speed_steer':
                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                front_obs_vec = np.array([1.5, 1.5])
                front_obs_vel = np.array([1.5, 1.5])
                front_min_dist = 10000

                front_right_obs_vec = np.array([1.5, 1.5])
                front_right_obs_vel = np.array([1.5, 1.5])
                front_right_min_dist = 10000

                front_left_obs_vec = np.array([1.5, 1.5])
                front_left_obs_vel = np.array([1.5, 1.5])
                front_left_min_dist = 10000

                back_right_obs_vec = np.array([1.5, 1.5])
                back_right_obs_vel = np.array([1.5, 1.5])
                back_right_min_dist = 10000

                back_left_obs_vec = np.array([1.5, 1.5])
                back_left_obs_vel = np.array([1.5, 1.5])
                back_left_min_dist = 10000


                for id, obstacle_data in agent.episode_measurements['obstacle_sensor']['state'].items():
                    # Compute dot product of obstacle vector with car vector
                    normalized_obstacle_vector = obstacle_data['position'] / np.linalg.norm(obstacle_data['position'])
                    # Dot product is simply the first element of the normalized vector
                    dot_product = normalized_obstacle_vector[0]

                    # Obstacle is in front of vehicle
                    if dot_product > 0.995 and obstacle_data['distance'] < front_min_dist:
                        front_min_dist = obstacle_data['distance']
                        front_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                        front_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in front right
                    elif dot_product > 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < front_right_min_dist:
                        front_right_min_dist = obstacle_data['distance']
                        front_right_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                        front_right_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in front left
                    elif dot_product > 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < front_left_min_dist:
                        front_left_min_dist = obstacle_data['distance']
                        front_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        front_left_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in back right
                    elif dot_product <= 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < back_right_min_dist:
                        back_right_min_dist = obstacle_data['distance']
                        back_right_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        back_right_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in back left
                    elif dot_product <= 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < back_left_min_dist:
                        back_left_min_dist = obstacle_data['distance']
                        back_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        back_left_obs_vel = obstacle_data['velocity'] / 20

                if(light != self.config['default_obs_traffic_val']):
                    unnorm_obs_dist = front_obs_vec[0] * self.config['vehicle_proximity_threshold']
                    unnorm_light = light * 20

                    # If the light is further do nothing
                    if(front_obs_vec[0] != self.config['default_obs_traffic_val'] and unnorm_light > unnorm_obs_dist):
                        pass
                    else:
                        front_obs_vec = np.array([light, 0]) / 20.0
                        front_obs_vel = np.array([0,0])


                # For visualization
                agent.episode_measurements['obstacle_dist'] = front_min_dist
                agent.episode_measurements['obstacle_speed'] = np.mean(np.square(front_obs_vel * 20))**0.5 
                agent.episode_measurements['obstacle_dist_front_right'] = front_right_min_dist
                agent.episode_measurements['obstacle_speed_front_right'] = np.mean(np.square(front_right_obs_vel * 20))**0.5 
                agent.episode_measurements['obstacle_dist_front_left'] = front_left_min_dist
                agent.episode_measurements['obstacle_speed_front_left'] = np.mean(np.square(front_left_obs_vel * 20))**0.5
                agent.episode_measurements['obstacle_dist_back_right'] = back_right_min_dist
                agent.episode_measurements['obstacle_speed_back_right'] = np.mean(np.square(back_right_obs_vel * 20))**0.5
                agent.episode_measurements['obstacle_dist_back_left'] = back_left_min_dist
                agent.episode_measurements['obstacle_speed_back_left'] = np.mean(np.square(back_left_obs_vel * 20))**0.5 

                obs['observation'] = np.concatenate(
                    (
                        np.array([agent.episode_measurements['next_orientation']]),
                        np.array([speed]),
                        np.array([steer]),
                        np.array([ldist]),
                        np.array([front_obs_vec[0]]),
                        np.array([front_obs_vec[1]]),
                        np.array([front_obs_vel[0]]),
                        np.array([front_obs_vel[1]]),
                        np.array([front_right_obs_vec[0]]),
                        np.array([front_right_obs_vec[1]]),
                        np.array([front_right_obs_vel[0]]),
                        np.array([front_right_obs_vel[1]]),
                        np.array([front_left_obs_vec[0]]),
                        np.array([front_left_obs_vec[1]]),
                        np.array([front_left_obs_vel[0]]),
                        np.array([front_left_obs_vel[1]]),
                        np.array([back_right_obs_vec[0]]),
                        np.array([back_right_obs_vec[1]]),
                        np.array([back_right_obs_vel[0]]),
                        np.array([back_right_obs_vel[1]]),
                        np.array([back_left_obs_vec[0]]),
                        np.array([back_left_obs_vec[1]]),
                        np.array([back_left_obs_vel[0]]),
                        np.array([back_left_obs_vel[1]]),
                    )
                )
            
            elif self.config['input_type'] == 'wp_360_obstacle_steer':
                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                front_obs_vec = np.array([1.5, 1.5])
                front_obs_vel = np.array([1.5, 1.5])
                front_min_dist = 10000

                front_right_obs_vec = np.array([1.5, 1.5])
                front_right_obs_vel = np.array([1.5, 1.5])
                front_right_min_dist = 10000

                front_left_obs_vec = np.array([1.5, 1.5])
                front_left_obs_vel = np.array([1.5, 1.5])
                front_left_min_dist = 10000

                back_right_obs_vec = np.array([1.5, 1.5])
                back_right_obs_vel = np.array([1.5, 1.5])
                back_right_min_dist = 10000

                back_left_obs_vec = np.array([1.5, 1.5])
                back_left_obs_vel = np.array([1.5, 1.5])
                back_left_min_dist = 10000


                for id, obstacle_data in agent.episode_measurements['obstacle_sensor']['state'].items():
                    # Compute dot product of obstacle vector with car vector
                    normalized_obstacle_vector = obstacle_data['position'] / np.linalg.norm(obstacle_data['position'])
                    # Dot product is simply the first element of the normalized vector
                    dot_product = normalized_obstacle_vector[0]

                    # Obstacle is in front of vehicle
                    if dot_product > 0.995 and obstacle_data['distance'] < front_min_dist:
                        front_min_dist = obstacle_data['distance']
                        front_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                        front_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in front right
                    elif dot_product > 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < front_right_min_dist:
                        front_right_min_dist = obstacle_data['distance']
                        front_right_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                        front_right_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in front left
                    elif dot_product > 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < front_left_min_dist:
                        front_left_min_dist = obstacle_data['distance']
                        front_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        front_left_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in back right
                    elif dot_product <= 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < back_right_min_dist:
                        back_right_min_dist = obstacle_data['distance']
                        back_right_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        back_right_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in back left
                    elif dot_product <= 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < back_left_min_dist:
                        back_left_min_dist = obstacle_data['distance']
                        back_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        back_left_obs_vel = obstacle_data['velocity'] / 20

                if(light != self.config['default_obs_traffic_val']):
                    unnorm_obs_dist = front_obs_vec[0] * self.config['vehicle_proximity_threshold']
                    unnorm_light = light * 20

                    # If the light is further do nothing
                    if(front_obs_vec[0] != self.config['default_obs_traffic_val'] and unnorm_light > unnorm_obs_dist):
                        pass
                    else:
                        front_obs_vec = np.array([light, 0]) / 20.0
                        front_obs_vel = np.array([0,0])


                # For visualization
                agent.episode_measurements['obstacle_dist'] = front_min_dist
                agent.episode_measurements['obstacle_speed'] = np.mean(np.square(front_obs_vel * 20))**0.5 
                agent.episode_measurements['obstacle_dist_front_right'] = front_right_min_dist
                agent.episode_measurements['obstacle_speed_front_right'] = np.mean(np.square(front_right_obs_vel * 20))**0.5 
                agent.episode_measurements['obstacle_dist_front_left'] = front_left_min_dist
                agent.episode_measurements['obstacle_speed_front_left'] = np.mean(np.square(front_left_obs_vel * 20))**0.5
                agent.episode_measurements['obstacle_dist_back_right'] = back_right_min_dist
                agent.episode_measurements['obstacle_speed_back_right'] = np.mean(np.square(back_right_obs_vel * 20))**0.5
                agent.episode_measurements['obstacle_dist_back_left'] = back_left_min_dist
                agent.episode_measurements['obstacle_speed_back_left'] = np.mean(np.square(back_left_obs_vel * 20))**0.5 

                obs['observation'] = np.concatenate(
                    (
                        np.array([agent.episode_measurements['next_orientation']]),
                        np.array([steer]),
                        np.array([ldist]),
                        np.array([front_obs_vec[0]]),
                        np.array([front_obs_vec[1]]),
                        np.array([front_obs_vel[0]]),
                        np.array([front_obs_vel[1]]),
                        np.array([front_right_obs_vec[0]]),
                        np.array([front_right_obs_vec[1]]),
                        np.array([front_right_obs_vel[0]]),
                        np.array([front_right_obs_vel[1]]),
                        np.array([front_left_obs_vec[0]]),
                        np.array([front_left_obs_vec[1]]),
                        np.array([front_left_obs_vel[0]]),
                        np.array([front_left_obs_vel[1]]),
                        np.array([back_right_obs_vec[0]]),
                        np.array([back_right_obs_vec[1]]),
                        np.array([back_right_obs_vel[0]]),
                        np.array([back_right_obs_vel[1]]),
                        np.array([back_left_obs_vec[0]]),
                        np.array([back_left_obs_vec[1]]),
                        np.array([back_left_obs_vel[0]]),
                        np.array([back_left_obs_vel[1]]),
                    )
                )
            
            elif self.config['input_type'] == 'wp_360_obstacle':
                speed = agent.episode_measurements['speed'] / 10
                steer = agent.episode_measurements['control_steer']
                ldist = agent.episode_measurements['dist_to_trajectory']
                light = agent.episode_measurements['red_light_dist']

                if light != -1:
                    light /= self.config['traffic_light_proximity_threshold']
                else:
                    light = self.config['default_obs_traffic_val']

                front_obs_vec = np.array([1.5, 1.5])
                front_obs_vel = np.array([1.5, 1.5])
                front_min_dist = 10000

                front_right_obs_vec = np.array([1.5, 1.5])
                front_right_obs_vel = np.array([1.5, 1.5])
                front_right_min_dist = 10000

                front_left_obs_vec = np.array([1.5, 1.5])
                front_left_obs_vel = np.array([1.5, 1.5])
                front_left_min_dist = 10000

                back_right_obs_vec = np.array([1.5, 1.5])
                back_right_obs_vel = np.array([1.5, 1.5])
                back_right_min_dist = 10000

                back_left_obs_vec = np.array([1.5, 1.5])
                back_left_obs_vel = np.array([1.5, 1.5])
                back_left_min_dist = 10000


                for id, obstacle_data in agent.episode_measurements['obstacle_sensor']['state'].items():
                    # Compute dot product of obstacle vector with car vector
                    normalized_obstacle_vector = obstacle_data['position'] / np.linalg.norm(obstacle_data['position'])
                    # Dot product is simply the first element of the normalized vector
                    dot_product = normalized_obstacle_vector[0]

                    # Obstacle is in front of vehicle
                    if dot_product > 0.995 and obstacle_data['distance'] < front_min_dist:
                        front_min_dist = obstacle_data['distance']
                        front_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                        front_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in front right
                    elif dot_product > 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < front_right_min_dist:
                        front_right_min_dist = obstacle_data['distance']
                        front_right_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                        front_right_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in front left
                    elif dot_product > 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < front_left_min_dist:
                        front_left_min_dist = obstacle_data['distance']
                        front_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        front_left_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in back right
                    elif dot_product <= 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < back_right_min_dist:
                        back_right_min_dist = obstacle_data['distance']
                        back_right_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        back_right_obs_vel = obstacle_data['velocity'] / 20

                    # Obstacle is in back left
                    elif dot_product <= 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < back_left_min_dist:
                        back_left_min_dist = obstacle_data['distance']
                        back_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                        back_left_obs_vel = obstacle_data['velocity'] / 20

                if(light != self.config['default_obs_traffic_val']):
                    unnorm_obs_dist = front_obs_vec[0] * self.config['vehicle_proximity_threshold']
                    unnorm_light = light * 20

                    # If the light is further do nothing
                    if(front_obs_vec[0] != self.config['default_obs_traffic_val'] and unnorm_light > unnorm_obs_dist):
                        pass
                    else:
                        front_obs_vec = np.array([light, 0]) / 20.0
                        front_obs_vel = np.array([0,0])


                # For visualization
                agent.episode_measurements['obstacle_dist'] = front_min_dist
                agent.episode_measurements['obstacle_speed'] = np.mean(np.square(front_obs_vel * 20))**0.5 
                agent.episode_measurements['obstacle_dist_front_right'] = front_right_min_dist
                agent.episode_measurements['obstacle_speed_front_right'] = np.mean(np.square(front_right_obs_vel * 20))**0.5 
                agent.episode_measurements['obstacle_dist_front_left'] = front_left_min_dist
                agent.episode_measurements['obstacle_speed_front_left'] = np.mean(np.square(front_left_obs_vel * 20))**0.5
                agent.episode_measurements['obstacle_dist_back_right'] = back_right_min_dist
                agent.episode_measurements['obstacle_speed_back_right'] = np.mean(np.square(back_right_obs_vel * 20))**0.5
                agent.episode_measurements['obstacle_dist_back_left'] = back_left_min_dist
                agent.episode_measurements['obstacle_speed_back_left'] = np.mean(np.square(back_left_obs_vel * 20))**0.5 

                obs['observation'] = np.concatenate(
                    (
                        np.array([agent.episode_measurements['next_orientation']]),
                        np.array([ldist]),
                        np.array([front_obs_vec[0]]),
                        np.array([front_obs_vec[1]]),
                        np.array([front_obs_vel[0]]),
                        np.array([front_obs_vel[1]]),
                        np.array([front_right_obs_vec[0]]),
                        np.array([front_right_obs_vec[1]]),
                        np.array([front_right_obs_vel[0]]),
                        np.array([front_right_obs_vel[1]]),
                        np.array([front_left_obs_vec[0]]),
                        np.array([front_left_obs_vec[1]]),
                        np.array([front_left_obs_vel[0]]),
                        np.array([front_left_obs_vel[1]]),
                        np.array([back_right_obs_vec[0]]),
                        np.array([back_right_obs_vec[1]]),
                        np.array([back_right_obs_vel[0]]),
                        np.array([back_right_obs_vel[1]]),
                        np.array([back_left_obs_vec[0]]),
                        np.array([back_left_obs_vec[1]]),
                        np.array([back_left_obs_vel[0]]),
                        np.array([back_left_obs_vel[1]]),
                    )
                )
            

    def fetch_symbolic_dict(self, ego_agent):
            # get ego kinematics
            ego_actor = ego_agent.vehicle_actor
            ego_features = env_util.fetch_actor_features(ego_actor)

            ref = ego_features['x'], ego_features['y']
            theta = ego_features['theta']

            env_util.normalize_actor_features(ego_features, ref, theta)

            # get other entities
            other_actors = self._world.get_actors().filter('*vehicle*')
            vehicle_features = {actor.id: env_util.fetch_actor_features(actor) for actor in other_actors
                if actor.get_transform().location.distance(ego_actor.get_transform().location) < 20
                and actor.id != ego_actor.id
            }

            for vehicle_id in vehicle_features:
                features = vehicle_features[vehicle_id]
                env_util.normalize_actor_features(features, ref, theta)

            # normalize waypoints
            # print(len(ego_agent.next_waypoints))
            waypoints = [
                (
                    wp.transform.location.x,
                    wp.transform.location.y,
                    wp.transform.location.z,
                )
                for wp in ego_agent.next_waypoints
            ]
            for i, (x,y,_) in enumerate(waypoints):
                x,y = transform_to_pov((x,y), ref, theta)
                waypoints[i] = (x,y)

            features = {
                'ego_features': ego_features,
                'vehicle_features': vehicle_features,

                'light': ego_agent.episode_measurements['red_light_dist'],

                'next_waypoints': waypoints,
                'next_orientation': ego_agent.episode_measurements['next_orientation'],
                'dist_to_trajectory': ego_agent.episode_measurements['dist_to_trajectory'],

                'obstacle_dist': ego_agent.episode_measurements['obstacle_dist'],
                'obstacle_speed': ego_agent.episode_measurements['obstacle_speed'],

                'x': ref[0],
                'y': ref[1],
                'theta': theta
            }
            return features

    def _update_straight_dynamic_obs(self, agent):
        car_spawn_point = Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))
        location = agent.vehicle_actor.get_location()
        distance_to_car = location.distance(car_spawn_point.location)

        agent.episode_measurements['obstacle_dist'] = distance_to_car

        if distance_to_car < 20:
            agent.episode_measurements['obstacle_visible'] = True
        else:
            agent.episode_measurements['obstacle_visible'] = False

    