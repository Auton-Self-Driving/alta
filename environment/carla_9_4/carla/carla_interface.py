from environment.carla_9_4.carla.server import CarlaServer
from environment.carla_9_4.carla.vehicle_manager import VehicleManager
from abc import ABC
import time


# TODO make sure carla import works
import carla

#TODO add handling for offline map - needed for leaderboard


class Carla910Interface:

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
        self.vehicle_manager = VehicleManager(self.config, self.world)

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        print("server_version", self.client.get_server_version())

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.server.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config["client_timeout_seconds"])

        return client






