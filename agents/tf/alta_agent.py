from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import numpy as np
from leaderboard.autoagents import models, controller
import carla


from environment.carla_9_4.env_util import (
    get_world_coords_from_latlong,
    convert_route_from_GPS_world
)

from environment.carla_9_4.planner import GlobalPlanner

import ipdb
st = ipdb.set_trace

def get_entry_point():
    return 'AltaAgent'

class AltaAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file):
        super(AltaAgent, self).__init__(path_to_conf_file)
        self.track = Track.MAP

        # Move this to configuration file later
        self.mode = "Imitation"
        self.image_type = "rgb"
        self.semantic_classes = 5
        self.pretrained_weights_path = None
        self.target_speed = 20
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}
        self.controller = controller.PIDLongitudinalController(K_P=self.args_longitudinal_dict['K_P'], K_D=self.args_longitudinal_dict['K_D'], K_I=self.args_longitudinal_dict['K_I'], dt=self.args_longitudinal_dict['dt'])

        image_params = self.sensors()[0]
        self.image_size = tuple([image_params['height'], image_params['width'], 3])
        if self.image_type=="semantic":
            self.image_size = tuple([image_params['height'], image_params['width'], self.semantic_classes])

        if self.mode=="Imitation":
            # z_size if for specifying how many manual states are being used
            self.policy_network = models.ConvPrImitator(z_size = 5, image_size = self.image_size, is_training=False, gpu_mode=True)
            if self.pretrained_weights_path:
                self.policy_network.load_json(self.pretrained_weights_path)
            else:
                self.policy_network.set_random_params()

        # Storing the OpenDRIVE MAP
        self._map = None
        self.global_planner = None

        # Initialize previous steer
        self.previous_steer = 0



    def setup(self, path_to_conf_file):
        self.track = Track.MAP # At a minimum, this method sets the Leaderboard modality. In this case, SENSORS

    # To be Modified if required
    def sensors(self):
        sensors = [{'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 300, 'height': 200, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -45.0, 'id': 'IMU'},
            {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'OpenDRIVE'},
           {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'SPEED'},
           ]
        return sensors

    def _configure_planner(self, map_string):
        # Instantiate the global planner
        self.planner = GlobalPlanner()
        self.trace_route = []
        for idx in range(len(self.scenario_route) - 1):
            source = self.scenario_route[idx]
            destination = self.scenario_route[idx+1]
            trace_route = self.global_planner._trace_route(self._map,
                            source, destination)
            self.trace_route.extend(trace_route)

        self.global_planner.set_global_plan(self.trace_route)



    def _preprocess_image(self, image):
        #array = np.reshape(array, (image.shape[0], image.shape[1], 4))
        image = image[:, :, :3]
        image = image[:, :, ::-1]
        return image

    def get_sematic_info(self, image):
        return None

    def get_traffic_light_info(self, image):
        return 0

    # def get_waypoint_info(self, vehicle_transform, map):


    def compute_wp_stats(self, vehicle_transform):
        "Return type: list containing [mean_angle, ldist, distance_to_goal_trajec]"
        mean_angle, ldist, distance_to_goal_trajec, _, _, _ = self.global_planner.get_next_orientation_new(vehicle_transform)

        return mean_angle, ldist, distance_to_goal_trajec

    def get_motion_info(self, imu, speedometer):
        "Return type: list containing [steer, speed]"
        return [0,0]

    def _get_vehicle_transform(self, gnss_reading, imu_reading):
        # Convert to x,y,z
        world_coords = get_world_coords_from_latlong(gnss_reading.latitude, gnss_reading.longitude, gnss_reading.altitude)

        x,y,z = world_coords[0][0], world_coords[1][0], world_coords[2][0]

        # Construct transform
        return carla.Transform(carla.Location(x = x, y = y, z = z), carla.Rotation(yaw = imu_reading.compass))

    def preprocess_inputs(self, input_data):
        # Configure planner when we first receive MAP info
        if(self.global_planner is None):
            self._configure_planner(input_data['OpenDRIVE'])

        processed_input = {}

        rgb_image = self._preprocess_image(input_data['Center'][1])
        processed_input['rgb'] = rgb_image

        # Audrey, Brian and Mayank
        semantic_image = self.get_sematic_info(rgb_image)
        processed_input['semantic'] = semantic_image

        # Zhe and Swapnil
        traffic_light = self.get_traffic_light_info(rgb_image)
        processed_input['tlight'] = traffic_light

        vehicle_transform = self._get_vehicle_transform(input_data["GPS"], input_data['IMU'])

        processed_input["mean_angle"], processed_input['ldist'], processed_input['distance_to_goal_trajec'] = compute_wp_stats(vehicle_transform)


        processed_input['steer'] = self.previous_steer
        processed_input['speed'] = input_data['SPEED']

        return processed_input

    def get_action(self, inputs, mode="Imitation"):
        mean_angle = inputs['mean_angle']
        ldist = inputs['ldist']
        distance_to_goal_trajec = inputs['distance_to_goal_trajec']

        obstacle_dist = 0
        obstacle_speed = 0

        steer = inputs['steer']
        speed = inputs['speed']

        light = inputs['tlight']

        low_dim_input = np.concatenate((np.array([mean_angle]), \
                                        np.array([obstacle_dist]), \
                                        np.array([obstacle_speed]), \
                                        np.array([speed]), \
                                        np.array([steer]), \
                                        np.array([ldist]), \
                                        np.array([distance_to_goal_trajec]), \
                                        np.array([light])))

        if mode=="Imitation":
            filtered_low_dim_input = np.concatenate([low_dim_input[:1], low_dim_input[3:5], low_dim_input[6:]])[None,:]
            if self.image_type=="rgb":
                img = np.expand_dims(inputs['rgb'], axis = 0)
            else:
                img = np.expand_dims(inputs['semantic'], axis = 0)

            action = self.policy_network.predict(img, filtered_low_dim_input)
        #TODO: Include policy networks other 2 modes

        return action

    def get_control(self, action, current_speed):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """
        steer = np.clip(float(action[0]), -1.0, 1.0)
        target_speed = (action[1] * 1.5) + 1
        target_speed = float(np.clip(target_speed * 10, 0, self.target_speed))

        gas = self.controller.pid_control(target_speed, current_speed, enable_brake=True)
        if gas < 0:
            throttle = 0.0
            brake = abs(gas)
        else:
            throttle = gas
            brake = 0.0

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return control

    def run_step(self, input_data, timestamp):
        preprocess_inputs = self.preprocess_inputs(input_data)
        print(preprocess_inputs)
        action = self.get_action(preprocess_inputs, mode=self.mode)

        control = self.get_control(action[0], preprocess_inputs['speed'])
        self.previous_steer = control.steer

        return control
