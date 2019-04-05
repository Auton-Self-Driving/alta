import random
import logging
import traceback
import math
import time
from scipy.misc import imresize
import torch

from carla.client import CarlaClient
from carla.planner.planner import Planner
from carla.settings import CarlaSettings
from carla.sensor import Camera
from carla.util import print_over_same_line
from carla.carla_server_pb2 import Control

from .server import CarlaServer


def make_carla_env(port):
    return lambda: CarlaEnv(port)


class CarlaEnv(object):
    def __init__(self, port=2000, city_name="Town01", frame_skip=1, dump_obs=False):
        """ 
        Inputs:
            - port: port at which to run Carla server and client
            - city_name: Carla city in which to run experiments
            - frame_skip: each action is executed skip_frame frames in a row
            - dump_obs: print measurements of vehicle movement and dump state 
                observations to disk
        """
        self.port = port
        self.frame_skip = frame_skip
        self.dist_for_success = 2.0
        self.dump_obs = dump_obs
        self.episode_ind = 0
        self.frame_ind = 0
        
        # Carla server
        self.server = CarlaServer(port)
        time.sleep(5)
        
        # Carla client
        trial = 0
        max_trials = 5
        while(trial < max_trials):
            try: 
                self.client = CarlaClient("localhost", port)
                self.client.connect()
                print("[Client connected] at port {}".format(port))
                break
            except Exception as e:
                trial += 1
                continue

        if trial == max_trials:
            print("[Client connection failed] at port {}".format(port))
            logging.error(traceback.format_exc())
        
        # Planner to predict high-level directions
        self._planner = Planner(city_name)
            
        # Weird Carla bug: first episode with a newly connected client does not 
        # load the right start and end positions, quick fix: start a dummy episode
        trial = 0
        max_trials = 5
        while(trial < max_trials):
            try:
                self._load_settings()
                self.client.start_episode(0)
                break
            except:
                trial += 1
                continue
        
        if trial == max_trials:
            print("[Dummy episode failed] at port {}".format(port))

    def reset(self):
        """ Reset simulator
        Output:
            - obs: dictionary with fields below 
                image: tensor of shape (1, 3, 88, 200)
                speed: tensor of shape (1, 1) 
                branch_mask: one-hot tensor of shape (1, 4)
                autopilot_action: tensor of shape (1, 2) 
        """
        self.episode_ind += 1
        self.frame_ind = 0
        
        # Load settings
        scene = self._load_settings()

        # Sample start and target positions
        # [36, 40], [39, 35]   Straight
        # [49, 18], [88, 28]   Right turn
        # [134, 52], [88, 151] Left turn
        start_idx, target_idx = [53, 67]
         
        self.start = scene.player_start_spots[start_idx]
        self.target = scene.player_start_spots[target_idx]
        
        # Keep track of conditions to stop episode
        self.offlane_steps = 0
        self.static_steps = 0
        
        # Start episode
        trial = 0
        max_trials = 5
        while(trial < max_trials):
            try: 
                self.client.start_episode(start_idx)
                obs = self._process_observation(*self._get_raw_observation())
                print("[Episode started] at port {}".format(self.port))
                break
            except Exception as e:
                trial += 1
                print("[Episode start failed] at port {} and trial {}".format(self.port, trial))
                continue
            
        if trial == max_trials:
            print("[Episode start failed] at port {}".format(self.port))
            logging.error(traceback.format_exc())
        
        # Get car moving
        while obs["speed"].item() < 0.1:
            control = Control()
            control.steer = 0
            control.throttle = 1
            control.brake = 0
            self.client.send_control(control)
            obs = self._process_observation(*self._get_raw_observation())

        return obs
    
    def _load_settings(self):
        settings = CarlaSettings()
        
        settings.set(NumberOfVehicles=0,
                     NumberOfPedestrians=0,
                     WeatherId=1)
        
        # Same camera settings as in CORL17 benchmark
        camera = Camera("CameraRGB")
        camera.set(FOV=100)
        camera.set_image_size(800, 600)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)
        settings.add_sensor(camera)
        
        # Load settings
        trial = 0
        max_trials = 5
        while(trial < max_trials):
            try:
                scene = self.client.load_settings(settings)
                print("[Loading settings] at port {}".format(self.port))
                break
            except Exception as e:
                trial += 1
                continue

        if trial == max_trials:
            print("[Settings loading failed] at port {}".format(self.port))
            logging.error(traceback.format_exc())
            
        return scene

    def step(self, action):
        """ Take a step in simulator
        Input:
            - action: tensor of shape (1, 2), steer and throttle/brake in [-1, 1]
        Outputs:
            - obs: dictionary with fields below 
                image: tensor of shape (1, 3, 88, 200)
                speed: tensor of shape (1, 1) 
                branch_mask: one-hot tensor of shape (1, 4)
                autopilot_action: tensor of shape (1, 2)
            - reward: tensor of shape (1, 1)
            - done: flag indicating if episode is over, tensor of shape (1, 1)
        """
        self.frame_ind += 1
        
        # Build Carla control object from action tensor
        control = self.get_control(action)
        
        # Take frame_skip steps in environment
        for _ in range(self.frame_skip - 1):
            self.client.send_control(control)
            self.client.read_data()
        self.client.send_control(control)
        
        # Get raw observation from simulator
        measurements, sensor_data, directions = self._get_raw_observation()
        
        if self.dump_obs:
            # Print measurements
            self._print_measurements(measurements)

            # Save images to disk
            self._save_episode_data(sensor_data)
            
        # Compute reward and done flag
        reward, done, info = self._get_reward(control, measurements, directions)
        
        # Extract observation tensors from raw observation
        obs = self._process_observation(measurements, sensor_data, directions)

        return obs, reward, done, info
    
    def _print_measurements(self, measurements):
        number_of_agents = len(measurements.non_player_agents)
        player_measurements = measurements.player_measurements
        message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
        message += '{speed:.0f} km/h, '
        message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
        message += '({agents_num:d} non-player agents in the scene)'
        message = message.format(
            pos_x=player_measurements.transform.location.x,
            pos_y=player_measurements.transform.location.y,
            speed=player_measurements.forward_speed * 3.6,  # m/s -> km/h
            col_cars=player_measurements.collision_vehicles,
            col_ped=player_measurements.collision_pedestrians,
            col_other=player_measurements.collision_other,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad,
            agents_num=number_of_agents)
        print_over_same_line(message)

    def _save_episode_data(self, sensor_data):
        out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
        for name, measurement in sensor_data.items():
            filename = out_filename_format.format(
                self.episode_ind, name, self.frame_ind)
            measurement.save_to_disk(filename)

    def _get_raw_observation(self):
        # Get measurements and image
        measurements, sensor_data = self.client.read_data()
             
        # Compute high-level directions with planner
        pos = measurements.player_measurements.transform
        directions = self._get_directions(pos)
        
        return measurements, sensor_data, directions
    
    def _get_directions(self, pos):
        targ = self.target
        directions = self._planner.get_next_command(
            (pos.location.x, pos.location.y, 0.22),
            (pos.orientation.x, pos.orientation.y, pos.orientation.z),
            (targ.location.x, targ.location.y, 0.22),
            (targ.orientation.x, targ.orientation.y, targ.orientation.z)
        )
        return directions
    
    def _get_reward(self, control, measurements, directions):
        stats = measurements.player_measurements
        
        info = {}
        info["steer"] = control.steer
        info["throttle"] = control.throttle
        info["brake"] = control.brake
        
        # 1) Abnormal steer penalty
        """
        if (control.steer > 0) and (directions == 3): 
            # Turn right when should go left
            steer_penalty = -15
        elif (control.steer < 0) and (directions == 4): 
            # Turn left when should go right
            steer_penalty = -15
        elif (abs(control.steer) > 0.2) and (directions in [0, 2, 5]):
            # Turn when should go straight
            # TODO: directions 0, 2 could mean follow lane that is turning
            steer_penalty = -20
        else:
            steer_penalty = 0
        """
        steer_penalty = 0
        info["steer_penalty"] = steer_penalty
        
        # 2) Collision penalty
        collision1 = stats.collision_vehicles > 0
        collision2 = stats.collision_pedestrians > 0
        collision3 = stats.collision_other > 0
        collision = (collision1 or collision2 or collision3)
        collision_penalty = -30 if collision else 0
        info["collision_penalty"] = collision_penalty
        
        # 3) Sidewalk and opposite lane overlap penalty
        otherlane = stats.intersection_otherlane
        offroad = stats.intersection_offroad
        lane_penalty = -30 if (otherlane or offroad) else 0
        info["lane_penalty"] = lane_penalty
            
        # 4) Speed reward (in km/h)
        speed = stats.forward_speed * 3.6 
        info["speed"] = speed
        if directions in [0, 2]:
            # If following lane or going straight, limit speed to 30km/h
            speed_reward = speed if (speed < 30) else (60 - speed)
        else:
            # If approaching intersection, limit speed to 20km/h
            speed_reward = speed if (speed < 20) else (40 - speed)
        if speed == 0:
            self.static_steps += 1
        info["speed_reward"] = speed_reward
            
        # Total reward (approximately scaled to [0, 1] range)
        reward = steer_penalty + collision_penalty + lane_penalty + speed_reward
        info["total_reward"] = reward
        reward = torch.tensor([[reward / 30.0]], dtype=torch.float32)
        
        if otherlane or offroad:
            self.offlane_steps += 1
        if speed <= 0:
            self.static_steps += 1
        
        # Episode termination conditions
        success = self._dist_to_target(stats.transform) < self.dist_for_success
        offlane = self.offlane_steps > 5
        static = self.static_steps > 20
        done = success or collision or offlane or static
        done = torch.tensor([[done]], dtype=torch.float32)

        if success:
            termination_state = 'success'
        elif collision:
            termination_state = 'collision'
        elif offlane:
            termination_state = 'offlane'
        elif static:
            termination_state = 'static'
        else:
            termination_state = 'none'

        info['termination_state'] = termination_state
        return reward, done, info
    
    def _dist_to_target(self, position):
        # Compute L2 flying distance from position to target
        x1 = position.location.x, position.location.y
        x2 = self.target.location.x, self.target.location.y
        return math.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)
    
    def _center_crop(self, image, size=(224, 224)):
        crop_height, crop_width = size
        height, width = image.shape
        start_w = width // 2 - (crop_width // 2)
        start_h = height // 2 - (crop_height // 2)
        
        return image[start_h:start_h+crop_height, start_w:start_w+crop_width]

    def _process_observation(self, measurements, sensor_data, directions):
        """ Extract image, speed and branch mask input tensors from raw observation
        Inputs:
            - measurements: Carla object containing speed float
            - sensor_data: Carla object containing rgb image of shape (600, 800, 3)
            - directions: integer in {0,2,3,4,5} output by planner
        Output:
            - obs: dictionary of tensors
                image: tensor of shape (1, 3, 88, 200)
                speed: tensor of shape (1, 1) 
                branch_mask: one-hot tensor of shape (1, 4)
        """
        image = sensor_data["CameraRGB"].data
        stats = measurements.player_measurements
        obs = {}

        # Image
        image = image[115:510, :] # Cut top and bottom
        # image = image[37:188, :]  # Cut top and bottom
        # image = imresize(image, (224, 224)) # Resize
        image = imresize(image, (88, 200))  # Resize
        # image = self._center_crop(image, size=(500, 500))
        # image = imresize(image, (224, 224)) # Resize
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1).unsqueeze(0) # Reshape
        obs["image"] = image / 255.0 # Normalize

        # Speed
        speed =  torch.tensor([[stats.forward_speed]], dtype=torch.float32)
        obs["speed"] = speed * 3.6 / 30 # Normalize by speed of 30km/h

        # Branch mask corresponding to directions
        if directions in [0, 2]: 
            # Follow lane (no intersections ahead)
            branch_idx = 0
        elif directions == 3: 
            # Turn left at next intersection
            branch_idx = 2
        elif directions == 4:
            # Turn right at next intersection
            branch_idx = 3
        else:
            # Go straight at next intersection
            branch_idx = 1

        branch_mask = torch.eye(4)[branch_idx]
        obs["branch_mask"] = branch_mask.unsqueeze(dim=0)
        
        # Low dimensional features
        pos = stats.transform
        norm = self._dist_to_target(pos)
        obs["dist_to_target"] = torch.tensor([norm])
        obs["simple"] = torch.tensor(
            [[obs["speed"].item(), 
              (self.target.location.x - pos.location.x) / norm,
              (self.target.location.y - pos.location.y) / norm,
              pos.orientation.x,
              pos.orientation.y]], 
            dtype=torch.float32
        )

        # Autopilot action
        if hasattr(stats.autopilot_control, "throttle"):
            gas = stats.autopilot_control.throttle
        else:
            gas = -stats.autopilot_control.brake

        obs["autopilot"] = torch.tensor([[stats.autopilot_control.steer, gas]])

        return obs

    def get_control(self, action):
        """ Get Control object for Carla from action tensor
        Input:
            - action: tensor of shape (1, 2), steer and throttle/brake in [-1, 1]
        Output: 
            - control: Control object for Carla
        """
        steer = action[0, 0].item()

        gas = action[0, 1].item()
        if gas < 0:
            throttle = 0
            brake = abs(gas)
        else:
            throttle = gas
            brake = 0

        # Avoid fake braking (from Codevilla conditional imitation learning code)
        # Needed for imitation learning agent to succeed on benchmarks, should not 
        # be used with RL agents
        #if (brake < 0.1) or (brake < acc):
        #    brake = 0.0

        control = Control()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = 0
        control.reverse = 0

        return control

    def close(self):
        self.client.disconnect()
        self.server.close()
        
    def __del__(self):
        self.close()
