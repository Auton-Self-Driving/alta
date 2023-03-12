from carla_environment.configs.config import (
    DefaultMainConfig
)

config = DefaultMainConfig()
# Populate the needed fields
config.populate_config(
    model_config = 'DPPOConfig',
    observation_config = "LowDimObservationConfig", # Use the 8-dim observation space
    action_config = "MergedSpeedScaledTanhSpeed50Config", # Use the tanh speed steer space
    reward_config = "Simple2RewardConfig", # Simple2 Rewards
    scenario_config = "NoCrashDenseTown01ConfigCustom", # Run NoCrashEmpty Scenarios
    testing = False, # Training, not testing
    carla_gpu = 0, 
    device_list = [1,2,3],
)


config.model_config.set_parameter(
    'model_ids', {"save_suffix":"test","checkpoint":"","ckpt_mode":""}
)
config.model_config.set_parameter(
    'infrastructure_settings', {"num_workers":1,"save_freq":1000}
)


print(vars(config.model_config))

print('success')