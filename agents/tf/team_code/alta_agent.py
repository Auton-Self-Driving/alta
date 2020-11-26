from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import numpy as np
import models, controller
import carla

from env_util import (
    get_world_coords_from_latlong,
    convert_route_from_GPS_world
)

from planner import GlobalPlanner

import yaml
import pickle
from scipy.interpolate import interp1d
import torch
from detectron2.config import CfgNode
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from AdelaiDet.tools.train_net import Trainer
from util import *
import vis_module
from PIL import Image
import tensorflow as tf
import time
import ipdb
st = ipdb.set_trace

# audrey imports 
from ae.controller import AEController
from rl_models import Policy_1_layer, Policy_2_layer, CustomPolicy1, CustomPolicy2
from ppo import PPO
import queue
import matplotlib.pyplot as plt 

tf.enable_eager_execution()

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
        #self.pretrained_weights_path = '/zfsauton2/home/vkadi/projects/alta/alta-logs/imitate_ppo/front_exp3_combined-data2_pretrained-comb1.json'
        #self.pretrained_weights_path = '/zfsauton/datasets/ArgoRL/mayank/front_dagger_iter_5.json'
        self.pretrained_weights_path = 'initializations/front_dagger_iter_5.json'
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

        print("#"*100, "Initializing policy network")
        if self.mode=="Imitation":
            # z_size if for specifying how many manual states are being used
            self.policy_network = models.ConvPrImitator(z_size = 5, image_size = self.image_size, is_training=False, gpu_mode=True)
            if self.pretrained_weights_path:
                self.policy_network.load_json(self.pretrained_weights_path)
            else:
                self.policy_network.set_random_params()
        elif self.mode == 'RL': 
            self.frame_stack = 3
            self.agent_model_path = '/zfsauton/datasets/ArgoRL/tanmaya_thesis_experiments/dynamic_actors/thesis_models/representationFS_I/algo_PPO_input_wp_vae_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.005_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_train_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.005_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_train_vae_epochs_10__clip_0.2__mb_10__runid_3/models/ppo2_weights15000000.zip' 
            self.ae_weights_path = '/zfsauton/datasets/ArgoRL/tanmaya_thesis_experiments/dynamic_actors/thesis_models/representationFS_I/algo_PPO_input_wp_vae_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.005_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_train_vae_epochs_10__clip_0.2__mb_10_/algo_PPO_input_wp_vae_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.005_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_train_vae_epochs_10__clip_0.2__mb_10__runid_3/ae_weights/ae_15000000'
            self.vae = AEController(image_size=(128, 128, 5), frame_stack=self.frame_stack)
            self.vae.load(self.ae_weights_path)
            self.policy_network = PPO.load(self.agent_model_path, None)

            self.stacked_observation_queue = queue.Queue(maxsize=self.frame_stack)
            self.vae_encoding_norm_factor = 10

        print("#"*100, "Initializing Sem seg")
        start = time.time()
        tf.keras.backend.clear_session()
        self.semantic_network = tf.keras.models.load_model('initializations/AdelaiDet_model/model.h5', custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform}, compile=False)
        print(time.time()-start)

        # Zhe: traffic lights detection model
        # print(os.getcwd())
        print("#"*100, "Initializing Traffic light network")
        with open('initializations/AdelaiDet_model/config.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            # model = Trainer.build_model(CfgNode(cfg))
            ckpt = torch.load('initializations/AdelaiDet_model/state_dict.pth', map_location=torch.device('cuda'))
            # ckpt = DetectionCheckpointer(model)
            # loaded = ckpt._load_file('../../AdelaiDet_model/model_final.pth')

        print("#"*100, "Initializing Interpolator")
        # with open('../../../AdelaiDet_model/interpolator.pkl', 'rb') as f:
        #     self.dist_interpolator = pickle.load(f)
        self.dist_interpolator = lambda area: 804 / (area + 1e-6) + 0.378
        self.traffic_light_detector = DefaultPredictor(CfgNode(cfg))
        # self.traffic_light_detector.model.load_state_dict(loaded['model']) # OpenCV BGR format image input expected
        self.traffic_light_detector.model.load_state_dict(ckpt) # OpenCV BGR format image input expected
        self.MAX_DISTANCE = 10 # or any value that matches the need of the agent
        self.NO_DISTANCE = 1 # or any value that matches the need of the agent

        # Storing the OpenDRIVE MAP
        self._map = None
        self.global_planner = None

        # Initialize previous steer
        self.previous_steer = 0

        #TODO: Include policy networks other 2 modes 

        '''SCRATCH_DIR = '/home/scratch/vkadi/'
        IMAGES_PATH = SCRATCH_DIR+'test_images/'
        VIDEO_PATH = SCRATCH_DIR+'test_videos/'
        self.vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, 1, videos=True)'''

        print("#"*100, "Setup finished")

        self.ctr = 0
    
    def destroy(self):
        '''del self.controller
        del self.policy_network
        del self.semantic_network
        del self.traffic_light_detector
        del self.dist_interpolator
        del self.global_planner
        del self.vis_wrapper'''
        del self._map

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def setup(self, path_to_conf_file):
        self.track = Track.MAP # At a minimum, this method sets the Leaderboard modality. In this case, SENSORS

    # To be Modified if required
    def sensors(self):
        sensors = [{'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 128, 'height': 128, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 512, 'height': 512, 'fov': 100, 'id': 'Center_high_res'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -90.0, 'id': 'IMU'}, 
            {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'OpenDRIVE'},
           {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'SPEED'},
           ]
        return sensors

    def _configure_planner(self, map_string):
        self._map = carla.Map("map", map_string)

        # Instantiate the global planner
        self.scenario_route = convert_route_from_GPS_world(self._global_plan, self._map)

        self.global_planner = GlobalPlanner()
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
        image = image[:, :, :3]     # BGR
        image = image[:, :, ::-1]   # RGB
        return image

    def get_sematic_info(self, image):
        inp = np.expand_dims(image.astype(np.float32), axis = 0)
        semantic_image = self.semantic_network.predict(inp)
        return semantic_image
    
    # def get_traffic_light_info_depredcated(self, image):
    #     image = image[:, :, ::-1].copy() # RGB -> BGR
    #     res = self.traffic_light_detector(image)
    #     if len(res['instances']) == 0: # no lights
    #         return 1 # Green Light, No Distance
    #     else:
    #         area = res['instances'].pred_boxes[0].area().item()
    #         color = res['instances'].pred_classes[0].item() # 0: Green, 1: Red
    #         score = res['instances'].scores[0].item()
    #         num_ins = len(res['instances'])
    #         if color == 1 and score > .667:
    #             dist_pred = max(0, self.dist_interpolator(area))
    #             print('detector Red, dist: {:.4f}, score: {:.4f}, num_ins: {}'.format(dist_pred, score, num_ins), flush=True)
    #             return dist_pred
    #     return 1

    def get_traffic_light_info(self, image):
        image = image[:, :, ::-1].copy() # RGB -> BGR
        res = self.traffic_light_detector(image)
        if len(res['instances']) == 0: # no lights
            return self.NO_DISTANCE # Green Light, No Distance
        else:
            area = res['instances'].pred_boxes.area().tolist()
            cls = res['instances'].pred_classes.tolist() # 0: Green, 1: Red, 2: Sign, 3: Car
            #  print(cls)
            avg_score = res['instances'].scores.mean().tolist()
            std_score = res['instances'].scores.std().tolist()
            #             score_thres = avg_score + std_score
            #             score_thres = 0
            score_thres = avg_score
            score = res['instances'].scores.tolist()
            num_ins = len(res['instances'])
            
            for _area, _cls, _score in zip(area, cls, score):
                # note, score has been already sorted from high to low.
                #                 print(_area, _cls, _score)
                if _score <= score_thres: break # possible backgrounds
                if _cls == 0: break # if Green comes before Red, pred Green.
                if _cls == 1:
                    if _score > score_thres:
                        # predict Red.
                        dist_pred = self.dist_interpolator(_area)
                        # dist_pred > threshold
                        if dist_pred > self.MAX_DISTANCE: return self.NO_DISTANCE
                        # output for debug
                        print('Detected Red Light, dist: {:.4f}, score: {:.4f}, num_ins: {}'.format(
                            dist_pred, _score, num_ins), flush=True)
                        return dist_pred / self.MAX_DISTANCE # normalize to (0, 1
                    
        return self.NO_DISTANCE

    def compute_wp_stats(self, vehicle_transform):
        # "Return type: list containing [mean_angle, ldist, distance_to_goal_trajec]"
        mean_angle, ldist, distance_to_goal_trajec, _, _, _ = self.global_planner.get_next_orientation_new(vehicle_transform)
        return mean_angle, ldist, distance_to_goal_trajec

    def get_motion_info(self, imu, speedometer):
        # "Return type: list containing [steer, speed]"
        return [0,0]

    def _get_vehicle_transform(self, gnss_reading, imu_reading):
        # Convert to x,y,z
        #world_coords = get_world_coords_from_latlong(gnss_reading.latitude, gnss_reading.longitude, gnss_reading.altitude)
        world_coords = get_world_coords_from_latlong(gnss_reading[0], gnss_reading[1], gnss_reading[2], self._map)

        x,y,z = world_coords[0][0], world_coords[1][0], world_coords[2][0]

        # Construct transform
        return carla.Transform(carla.Location(x = x, y = y, z = z), carla.Rotation(yaw = imu_reading[-1]))

    def _add_to_stacked_queue(self, object_queue, object_to_add):

        assert (object_queue is not None and object_to_add is not None)

        if object_queue.full():
            # Pop out earlier stacked frame if queue is full
            object_queue.get()
        object_queue.put(object_to_add)

    def preprocess_inputs(self, input_data):
        input_data['IMU'][1][-1] = (input_data['IMU'][1][-1]*(180/np.pi))
        if input_data['IMU'][1][-1]>180:
            input_data['IMU'][1][-1] = input_data['IMU'][1][-1]-360

        # Configure planner when we first receive MAP info
        if(self.global_planner is None):
            self._configure_planner(input_data['OpenDRIVE'][1]['opendrive'])        

        processed_input = {}

        rgb_image = self._preprocess_image(input_data['Center'][1])
        processed_input['rgb'] = rgb_image

        # Audrey, Brian and Mayank
        semantic_image = self.get_sematic_info(rgb_image)
        processed_input['semantic'] = semantic_image

        if self.mode=="RL":
            if self.stacked_observation_queue.empty(): 
                for _ in range(self.frame_stack): 
                    self._add_to_stacked_queue(self.stacked_observation_queue, semantic_image)
            else: 
                self._add_to_stacked_queue(self.stacked_observation_queue, semantic_image)

        # Zhe and Swapnil
        #print("*"*50, "preprocessing high res rgb")        
        high_res_rgb = self._preprocess_image(input_data['Center_high_res'][1])
        high_res_rgb = high_res_rgb.astype(np.float32)

        traffic_light = self.get_traffic_light_info(high_res_rgb)
        #traffic_light = self.get_traffic_light_info(rgb_image)
        processed_input['tlight'] = traffic_light


        vehicle_transform = self._get_vehicle_transform(input_data["GPS"][1], input_data['IMU'][1])
        '''print("*"*50)
        print(input_data['IMU'][1][-1])
        print(vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z, vehicle_transform.rotation.yaw)
        print("*"*50)'''
        processed_input["mean_angle"], processed_input['ldist'], processed_input['distance_to_goal_trajec'] = self.compute_wp_stats(vehicle_transform)
        processed_input['distance_to_goal_trajec'] = processed_input['distance_to_goal_trajec']/500 # to match env.py preproc
        
        processed_input['steer'] = self.previous_steer
        processed_input['speed'] = input_data['SPEED'][1]['speed']

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
            if self.image_type=='rgb':
                img = np.expand_dims(inputs['rgb'], axis = 0)
                concat_vis = Image.fromarray(inputs['rgb'], 'RGB').convert('RGBA')
                img = img/255.0
            else:
                semantic_image_np = inputs['semantic']
                #semantic_image_np = tf.keras.backend.eval(inputs['semantic'])

                semantic_vis = convert_to_rgb(convert_from_one_hot(semantic_image_np[0]), reduced_classes=True).astype(np.uint8)
                semantic_vis_pil = Image.fromarray(semantic_vis, 'RGB').convert('RGBA')

                rgb_vis = Image.fromarray(inputs['rgb'], 'RGB').convert('RGBA')
                concat_vis = self.get_concat_h(rgb_vis, semantic_vis_pil)

                img = semantic_image_np
            #self.vis_wrapper.save_image(concat_vis, 1)

            filtered_low_dim_input = np.concatenate([low_dim_input[:1], low_dim_input[3:5], low_dim_input[6:]])[None,:]
            action = self.policy_network.predict(img, filtered_low_dim_input)
        #TODO: Include policy networks other 2 modes 

        elif mode == 'RL': 
            stacked_observation = np.concatenate(list(self.stacked_observation_queue.queue), axis=-1) #np.stack(list(self.stacked_observation_queue.queue), axis=2)
            visual_observation = self.vae.encode(stacked_observation[0])
            visual_observation = visual_observation / self.vae_encoding_norm_factor

            filtered_low_dim_input = np.concatenate([low_dim_input[:1], low_dim_input[3:]]).reshape([1, 6])
            fused_input = np.hstack([visual_observation, filtered_low_dim_input])
            action = self.policy_network.predict(fused_input, deterministic=True)

        return action

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def get_control(self, input_data, action):
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
        #current_speed = self.get_speed_from_velocity(input_data['SPEED'][1]['speed']) * 3.6
        current_speed = input_data['SPEED'][1]['speed']*3.6


        gas = self.controller.pid_control(target_speed, current_speed, enable_brake=True)
        if gas < 0:
            throttle = 0.0
            brake = abs(gas)
        else:
            throttle = gas
            brake = 0.0
        '''print("#"*50)
        print(current_speed, target_speed)
        print(throttle)
        print(brake)
        print(steer)
        print("#"*50)'''

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
        self.ctr+=1
        preprocess_inputs = self.preprocess_inputs(input_data)
        print([preprocess_inputs['mean_angle'], preprocess_inputs['ldist'], preprocess_inputs['distance_to_goal_trajec'], preprocess_inputs['steer'], preprocess_inputs['speed']])
        # if(self.ctr%200==0):
        #     st()
        action = self.get_action(preprocess_inputs, mode=self.mode)
        # print(action[0])
        control = self.get_control(input_data, action[0])
        return control