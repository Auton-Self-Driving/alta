from environment.carla_9_4.carla.common import CarlaInterface

from environment.carla_9_4.carla.server import CarlaServer
from environment.carla_9_4 import planner
from environment.carla_9_4.carla.actor_manager import ActorManager910
from abc import ABC
import time


# TODO make sure carla import works
import carla

#TODO add handling for offline map - needed for leaderboard


class Carla910Interface(CarlaInterface):

    def __init__(self, config):
        self.config = config

        # Instantiate and start server
        self.server = CarlaServer(config)

        self.client = None

    def setup(self):
        # Start the carla server and get a client
        self.server.start()
        self.client = self._spawn_client()

        # Get the world
        self.world = self.client.get_world()

        # Update the settings from the config
        settings = self.world.get_settings()
        if(self.config['sync_mode']):
            settings.synchronous_mode = True
        if self.config["server_fps"] is not None and self.config["server_fps"] != 0:
            settings.fixed_delta_seconds =  1.0 / float(self.config["server_fps"])

        # Enable rendering
        settings.no_rendering_mode = False

        self.world.apply_settings(settings)

        # Sleep to allow for settings to update
        time.sleep(20)

        # Retrieve map
        self.map = self.world.get_map()

        # Get blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Instantiate a vehicle manager to handle other actors
        self.actor_fleet = None

        #TODO Decide where this should be
        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        print("server_version", self.client.get_server_version())

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.server.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config["client_timeout_seconds"])

        return client


    def reset(self):
        ### Delete old actors
        self.actor_fleet.destroy_actors()


        self.actor_fleet = ActorManager910(self.config, self.client)


        ### Spawn new actors
        #TODO should we spawn here or later
        self.actor_fleet.spawn()

        # Tick for 15 frames to handle car initialization in air
        for _ in range(15):
            world_frame = self.world.tick()

        # Create a global planner to generate dense waypoints along route
        self.global_planner = planner.GlobalPlanner()


        ### Setup the global planner
        #TODO Move these two steps to the global planner if dense_waypoints not used later
        self.dense_waypoints  = self.global_planner._trace_route(self._map,
                                self.source_transform, self.destination_transform)

        self.global_planner.set_global_plan(self.trace_route)


        next_orientation, self.dist_to_trajectory, distance_to_goal_trajec, self.next_waypoints, self.next_wp_angles, self.next_wp_vectors = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings()

        #TODO Combine sensor_readings with ep_measurements
        ep_measurements = {
            'next_orientation' : next_orientation,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory
        }

        sensor_readings.update(ep_measurements)
        return sensor_readings

    def step(self, action):
        ep_measurement = self.actor_fleet.step(action)

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings()
        location = self.actor_fleet.ego_vehicle._vehicle.get_location()

        sensor_readings["location"] = location

        world_frame = self.world.tick()

        return sensor_readings, ep_measurement




    def destroy_all_actors(self):
        # raise NotImplementedError()
        pass








