from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import numpy as np
from leaderboard.autoagents import models, controller
import carla

import yaml
import pickle
from scipy.interpolate import interp1d
import torch
from detectron2.config import CfgNode
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from AdelaiDet.tools.train_net import Trainer

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

        # Zhe: traffic lights detection model
        # print(os.getcwd())
        with open('../../AdelaiDet_model/config.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            model = Trainer.build_model(CfgNode(cfg))
            ckpt = torch.load('../../AdelaiDet_model/state_dict.pth', map_location=torch.device('cuda'))
            # ckpt = DetectionCheckpointer(model)
            # loaded = ckpt._load_file('../../AdelaiDet_model/model_final.pth')
        with open('../../AdelaiDet_model/interpolator.pkl', 'rb') as f:
            self.dist_interpolator = pickle.load(f)
        self.traffic_light_detector = DefaultPredictor(CfgNode(cfg))
        # self.traffic_light_detector.model.load_state_dict(loaded['model']) # OpenCV BGR format image input expected
        self.traffic_light_detector.model.load_state_dict(ckpt) # OpenCV BGR format image input expected

        #TODO: Include policy networks other 2 modes 

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

    def _preprocess_image(self, image):
        #array = np.reshape(array, (image.shape[0], image.shape[1], 4))
        image = image[:, :, :3]
        image = image[:, :, ::-1]
        return image

    def get_sematic_info(self, image):
        return None
    
    def get_traffic_light_info(self, image):
        image = image[:, :, ::-1] # RGB -> BGR
        res = self.traffic_light_detector(image)
        if len(res['instances']) == 0: # no lights
            return None # Green Light, No Distance
        else:
            area = res['instances'].pred_boxes[0].area().item()
            color = res['instances'].pred_classes[0].item() # 0: Green, 1: Red
            score = res['instances'].scores[0].item()
            num_ins = len(res['instances'])
            if color == 1 and score > .667:
                dist_pred = max(0, self.dist_interpolator(area))
                print('detector Red, dist: {:.4f}, score: {:.4f}, num_ins: {}'.format(dist_pred, score, num_ins), flush=True)
                return dist_pred
        return None

    def get_waypoint_info(self, map):
        return None

    def get_motion_info(self, imu, speedometer):
        "Return type: list containing [steer, speed]"
        return [0,0]

    def preprocess_inputs(self, input_data):
        processed_input = {}

        rgb_image = self._preprocess_image(input_data['Center'][1])
        processed_input['rgb'] = rgb_image

        # Audrey, Brian and Mayank
        semantic_image = self.get_sematic_info(rgb_image)
        processed_input['semantic'] = semantic_image

        # Zhe and Swapnil
        traffic_light = self.get_traffic_light_info(rgb_image)
        processed_input['tlight'] = traffic_light

        # Hitesh and Tanmay
        #dense_waypoints = self.get_waypoint_info(input_data['OpenDRIVE'][1])
        processed_input['dense_wp'] = None

        # Who?
        steer_speed = self.get_motion_info(input_data['IMU'][1], input_data['SPEED'][1])
        processed_input['steer_speed'] = steer_speed

        return processed_input

    def compute_wp_stats(self, wp):
        "Return type: list containing [mean_angle, ldist, distance_to_goal_trajec]"
        return [0,0,0]

    def get_action(self, inputs, mode="Imitation"):
        mean_angle, ldist, distance_to_goal_trajec = self.compute_wp_stats(inputs['dense_wp'])

        obstacle_dist = 0
        obstacle_speed = 0
        
        steer, speed = inputs['steer_speed']

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

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """
        steer = np.clip(float(action[0]), -1.0, 1.0)
        target_speed = (action[1] * 1.5) + 1
        target_speed = float(np.clip(target_speed * 10, 0, self.target_speed))

        # TODO: Need to replace this once we get to know how to extract agent's current velocity from IMU/speedometer sensors
        #current_speed = self.get_speed_from_velocity(action[1]) * 3.6
        current_speed = action[1] * 3.6

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
        action = self.get_action(preprocess_inputs, mode=self.mode)
        control = self.get_control(action[0])

        return control
