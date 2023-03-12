from carla_environment.configs.base_config import BaseConfig

class BaseModelConfig(BaseConfig):

    def __init__(self):
       
        # Max number of steps to train for
        self.max_num_steps = None

        # Stores model identifiers as a dict
        # 1. save_suffix : Model Name
        # 2. checkpoint:Checkpoint to resume from
        # 3. skpt_mode ['','load',resume']
        #       '' - > start from scratch
        #       load -> only load model weights
        #       resume -> load weights and all hyperparameters
        self.model_ids = {
            "save_suffix": None,
            "checkpoint":None,
            "ckpt_mode":None
        }
        
        # Store network hyperparameters as a dict
        self.network_settings = None

        # Store training infra hyperparameters as a dict
        # Use this to specify distributed training settings
        self.infrastructure_settings = None

    def verify(self, ignore_keys = []):
        # Ignore keys contains keys we want to skip during verification

        parameters = vars(self)

        for name, value in parameters.items():
            # Check that value is not None
            # Raise Exception if value is None
            if (name not in ignore_keys) and (value is None):
                raise Exception("Missing value for parameter {} in config {}".format(
                        name,
                        self.__class__.__name__
                ))

            # If object is a dict, verify that no value is None within
            if isinstance(value, dict):
                for k in value:
                    if value[k] is None:
                        raise Exception("Missing value for dictionary parameter {} in key {} with in config {}".format(
                                name,
                                k,
                                self.__class__.__name__
                        ))


        print("Verified config {}. Note: this just checks for missing values!".format(self.__class__.__name__))


class DPPOConfig(BaseModelConfig):

    def __init__(self):

        super().__init__()
        
        self.max_num_steps = 30000000

        self.network_settings = {
            "gamma":0.99,
            "policy_lr":4e-4,
            "eps_clip": .2,
            "grad_clip":.5,
            "squash":True, # This enables tanh squashing of sampled actions from policy
            'focal_loss':False,
            'push_grad': False,
            'standard': False, # if False, will push traj after finishing an episode
        }

        self.infrastructure_settings = {
            'num_workers': 14,
            'num_servers': 1, # currently only support 1 server
            'num_threads_per_server': 1,
            'worker_grad_update_freq': 20000,
            'worker_optim_epochs': 10,
            'server_glb_update_freq': 100,
            'server_adaptive_freq': True,
            'save_freq': 300000,
        }

