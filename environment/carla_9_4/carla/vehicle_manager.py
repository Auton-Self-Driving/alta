import numpy as np
import os
class VehicleManager910():
    def __init__(self, config, world):
        self.config = config
        self.world = world

        # tm is valid for carla0.9.10. If using carla0.9.6, this has to be commented out
        self.tm = self.client.get_trafficmanager(4050)
        self.tm.set_synchronous_mode(True)

        # Only randomize order of spawn points if testing
        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(self.config.log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        # Get blueprints
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        if self.config["disable_two_wheeler"]:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]