from carla_environment.configs.base_config import BaseConfig



class BaseRewardConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Speed reward coefficient
        self.speed_coeff = None

        # Acceleration reward coefficient
        self.acceleration_coeff = None

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = None

        # Penalty for collision
        self.const_collision_penalty = None

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = None

        # Penalty for exiting the lane
        self.const_out_of_lane_penalty = None

        # Penalty for leaving the lane proportional to speed
        self.out_of_lane_penalty_speed_coeff = None

        # Penalty for red light violation
        self.const_light_penalty = None

        # Penalty for red light infraction proportional to speed
        self.light_penalty_speed_coeff = None

        # Penalty for steer reward
        self.steer_penalty_coeff =  None

        # Reward for success completion of trajectory
        self.success_reward = None

        # Constant reward given at every time step
        self.constant_positive_reward = None

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = None

        # Flag to enable reward clipping
        self.clip_reward = None

        # Flag to terminate episode on lane invasion
        self.enable_lane_invasion_termination = None

        # Flag to count lane invasion as collision
        self.enable_lane_invasion_collision = None

        # Penalty if agent remains static for longer than threshold
        self.static_penalty = None

class Simple2RewardConfig(BaseRewardConfig):
    def __init__(self):
        # Speed reward coefficient
        self.speed_coeff = 1

        # Acceleration reward coefficient
        self.acceleration_coeff = 0

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = 1

        # Penalty for collision
        self.const_collision_penalty = 250

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = 0

        # Penalty for exiting the lane
        self.const_out_of_lane_penalty = 250

        self.out_of_lane_penalty_speed_coeff = 0

        # Penalty for red light violation
        self.const_light_penalty = 250

        # Penalty for red light infraction proportional to speed
        self.light_penalty_speed_coeff = 250

        # Penalty for steer reward
        self.steer_penalty_coeff =  0

        # Reward for success completion of trajectory
        self.success_reward = 0

        # Constant reward given at every time step
        self.constant_positive_reward = 0

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = 1

        # Flag to enable reward clipping
        self.clip_reward = False

        # Flag to terminate episode on lane invasion
        self.enable_lane_invasion_termination = False

        # Flag to count lane invasion as collision
        self.enable_lane_invasion_collision = False

        # Penalty if agent remains static for longer than threshold
        self.static_penalty = 0

class NoOutOfLanePenaltyConfig(BaseRewardConfig):
    def __init__(self):
        # Speed reward coefficient
        self.speed_coeff = 1

        # Acceleration reward coefficient
        self.acceleration_coeff = 0

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = 1

        # Penalty for collision
        self.const_collision_penalty = 250

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = 0

        # Penalty for exiting the lane
        self.const_out_of_lane_penalty = 0

        # Penalty for collision proportional to speed
        self.out_of_lane_penalty_speed_coeff = 0

        # Penalty for red light violation
        self.const_light_penalty = 250

        # Penalty for red light infraction proportional to speed
        self.light_penalty_speed_coeff = 0

        # Penalty for steer reward
        self.steer_penalty_coeff =  0

        # Reward for success completion of trajectory
        self.success_reward = 0

        # Constant reward given at every time step
        self.constant_positive_reward = 0

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = 1

class NoTrafficLightConfig(BaseRewardConfig):
    def __init__(self):
        # Speed reward coefficient
        self.speed_coeff = 1

        # Acceleration reward coefficient
        self.acceleration_coeff = 0

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = 1

        # Penalty for collision
        self.const_collision_penalty = 250

        # Penalty for collision proportional to speed
        self.collision_penalty_speed_coeff = 250

        # Penalty for exiting the lane
        self.const_out_of_lane_penalty = 250

        self.out_of_lane_penalty_speed_coeff = 0

        # Penalty for red light violation
        # change here
        self.const_light_penalty = 0

        # Penalty for red light infraction proportional to speed
        # change here
        self.light_penalty_speed_coeff = 0

        # Penalty for steer reward
        self.steer_penalty_coeff =  0

        # Reward for success completion of trajectory
        self.success_reward = 0

        # Constant reward given at every time step
        self.constant_positive_reward = 0

        # Factor to normalize rewards (reward is divided by this value)
        self.reward_normalize_factor = 1
