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
            self.config["reward_function"] = "simple"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "throttle_only"
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["scenarios"] = "straight"
            self.config["x_res"] = 84
            self.config["y_res"] = 84
