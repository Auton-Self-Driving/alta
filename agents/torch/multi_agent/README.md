### Adding new state space
1. Update `train_dppo_run.get_state_action_dims()`
2. Add bounds to `carla_env._setup_observation_and_action_space()`
3. Insert construction logic into `carla_env.create_observation()`
4. Add Obstacle sensor reinitialization logic in `carla_env.reset_vehicle_agent()`
5. Add state space name to `carla_env._get_ego_input()` for parsing observations
