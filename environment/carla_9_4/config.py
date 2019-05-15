class ConfigManager(object):
    def __init__(self, algo='DDPG', scenarios="straight", action_type="merged_gas", reward="new", use_segmented=False):
        self.config = {}

        self._initialize_config(algo, scenarios, action_type, reward, use_segmented)
    
    def _initialize_config(self, algo, scenarios, action_type, reward, use_segmented):
        if algo == 'DDPG':
            self.config["algo"] = "DDPG"
            self.config["x_res"] = 200
            self.config["y_res"] = 88
            self.config["reward_function"] = reward
            self.config["discrete_actions"] = False
            self.config["train_config"] = "torch"
            self.config["action_type"] = action_type
            self.config["framestack"] = 1
            self.config["preprocess_crop_image"] = True
            self.config["max_static_steps"] = 1000
            self.config["grayscale"] = False
            self.config["verbose"] = True
            self.config["segmented"] = use_segmented
            self.config["scenarios"] = scenarios
        elif algo == 'DQN':
            self.config["algo"] = "DQN"
            self.config["x_res"] = 84
            self.config["y_res"] = 84
            self.config["reward_function"] = "new"
            self.config["discrete_actions"] = True
            self.config["train_config"] = "baselines"
            self.config["action_type"] = "sep_gas"
            self.config["framestack"] = 4
            self.config["grayscale"] = True
            self.config["segmented"] = use_segme
