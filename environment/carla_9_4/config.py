class ConfigManager(object):
    def __init__(self, algo='DDPG', action_type="merged_gas"):
        self.config = {}

        self._initialize_config(algo, action_type)
    
    def _initialize_config(self, algo, action_type):
        if algo == 'DDPG':
            self.config["algo"] = "DDPG"
            self.config["x_res"] = 200
            self.config["y_res"] = 84
            self.config["reward_function"] = "cirl"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "torch"
            self.config["action_type"] = action_type
        elif algo == 'DQN':
            self.config["algo"] = "DQN"
            self.config["x_res"] = 84
            self.config["y_res"] = 84
            self.config["reward_function"] = "corl"
            self.config["discrete_actions"] = True
            self.config["train_config"] = "baselines"
            self.config["action_type"] = "sep_gas"


