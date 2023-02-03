import numpy as np

class StatsLogger:


    def __init__(self, mode="side_obs_sensors"):

        self.__setup_logger(mode)
        self.mode = mode


    def __setup_logger(self, mode):

        if mode == "side_obs_sensors":
            self.config = {
                "obstacle_dist_front_right":[],"obstacle_speed_front_right":[],
                "obstacle_dist_back_right":[],"obstacle_speed_back_right":[],
                "obstacle_dist_front_left":[],"obstacle_speed_front_left":[],
                "obstacle_dist_back_left":[],"obstacle_speed_back_left":[],
                }
        else:
            raise Exception("Invalid Agent Logging Mode")


    def update(self, agent):

        if self.mode == "side_obs_sensors":

            for k in list(self.config.keys()):
                self.config[k].append(agent.episode_measurements[k])

    def retrieve_stats_in_numpy(self):

        new_dict = {}

        for k in list(self.config.keys()):

            # Duplicating last entry to reconcile dims with other stats logged within carla
            new_dict[k] = np.asarray(self.config[k]+[self.config[k][-1]])  

        return new_dict

    
    def reset(self):

        self.__setup_logger(self.mode)
