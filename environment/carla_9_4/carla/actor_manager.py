import numpy as np
import os
import random

# Need to change the imports to contain env flag
import environment.carla_9_4.scenarios as scenarios
from environment.carla_9_4.agents.navigation.agent import Agent
import environment.carla_9_4.sensors as sensors
# Need to change the imports to contain env flag

from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation

class ActorManager910():
    def __init__(self, config, client):
        '''
        Manages ego vehicle, other actors and sensors
        Assumes that sensormanager is always attached to ego vehicle

        Common/High level attributes are:
        1) Spawn points (Used for spwaning actors and also by planner)
        2) Blueprints
        '''
        self.config = config
        self.world = client.get_world()

        ################################################
        # Spawn points
        ################################################
        self.spawn_points = self.world.get_map().get_spawn_points()
        # Only randomize order of spawn points if testing
        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(self.config.log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        ################################################
        # Blueprints
        ################################################
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if self.config["disable_two_wheeler"]:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]


        # tm is valid for carla0.9.10. If using carla0.9.6, this has to be commented out
        # This is for autopilot purpose on npcs
        # push it to spawn_npc() function?
        self.tm = client.get_trafficmanager(4050)
        self.tm.set_synchronous_mode(True)

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")
        self.actor_list = []

    def spawn():
        self.ego_vehicle = self.spawn_ego_vehicle()
        self.sensor_manager = self.spawn_sensors()
        # Check how to obtain the function argument value of 'unseen' variable
        if self.config["sample_npc"]:
            number_of_vehicles = np.random.randint(low=self.config["num_npc_lower_threshold"], high=self.config["num_npc_upper_threshold"])
        else:
            number_of_vehicles = self.config["num_npc"]

        self.spawn_npc(number_of_vehicles, unseen)

    def spawn_ego_vehicle(self):
        '''
        Spawns and return ego vehicle/Agent
        '''
        # Spawn the actor
        # Create an Agent object with that actor
        # Return the agent instance
        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))


        # Spawning vehicle actor with retry logic as it fails to spawn sometimes
        NUM_RETRIES = 5
        for _ in range(NUM_RETRIES):
            # Need to check about passing source_transform
            self.vehicle_actor = self._world.try_spawn_actor(vehicle_bp, self.source_transform)
            if self.vehicle_actor is not None:
                break
            else:
                print("Unable to spawn vehicle actor at {0}, {1}.".format(self.source_transform.location.x, self.source_transform.location.y))
                print("Number of existing actors, {0}".format(len(self.actor_list)))
                self.destroy_actors()              # Do we need this as ego vehicle is the first one to be spawned?
                time.sleep(120)

        if self.vehicle_actor is not None:
            self.actor_list.append(self.vehicle_actor)
            # Need to move this variable to carla interface file
            self.location = self.vehicle_actor.get_location()
        else:
            raise Exception("Failed in spawning vehicle actor.")

        # Agent uses proximity_threshold to detect traffic lights.
        # Hence we use traffic_light_proximity_threshold while creating an Agent.
        vehicle_agent = Agent(self.vehicle_actor, self.config['traffic_light_proximity_threshold'])
        return vehicle_agent
    
    def spawn_sensors(self):
        if self.ego_vehicle is None:
            print("Not spwaning sensors as the parent actor is not initialized properly")
            return None
        sensor_manager = sensors.SensorManager(self.config, self.ego_vehicle)
        for k,v in self.sensor_manager.sensors.items():
            self.actor_list.append(v)
        return sensor_manager

    def spawn_npc(self, number_of_vehicles, unseen):
        npc_spawn_points = self.pick_npc_spawn_points(number_of_vehicles, unseen)
        count = number_of_vehicles
        for spawn_point in npc_spawn_points:
            #$
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

    def pick_npc_spawn_points(self, number_of_vehicles, unseen):
        if self.config["scenarios"] == "straight_dynamic":
            # vehicle spawn_points corresponding to 84, 40
            spawn_points = [Transform(Location(x=-2.4200193881988525, y=187.97000122070312, z=1.32), Rotation(yaw=89.9996109008789)),
                        Transform(Location(x=1.5599803924560547, y=187.9700164794922, z=1.32), Rotation(yaw=-90.00040435791016))]

            # vehicle spawn_points corresponding to 96, 140
            # spawn_points = [Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
            # Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))]
        elif self.config["scenarios"] == "crowded":
            spawn_points = scenarios.get_crowded_npcs(number_of_vehicles)
            print('CROWDED SPAWNING: ', spawn_points)
        elif self.config["scenarios"] in ["long_straight", "long_straight_junction"]:
            spawn_points_1 = scenarios.get_long_straight_npcs()
            if unseen:
                if self.config["test_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config["train_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)

        elif self.config["scenarios"] == "straight_crowded":
            spawn_points = scenarios.get_straight_crowded_npcs(number_of_vehicles)
            print('STRAIGHT CROWDED SPAWNING: ', spawn_points)
        elif self.config["scenarios"] == "town3":
            spawn_points = scenarios.get_curved_town03_npcs(number_of_vehicles)
            print('TOWN 3 SPAWNING: ', spawn_points)

        else:
            # Testing
            if unseen:
                if self.config["test_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config["train_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)


        if self.config["verbose"]:
            print('found %d spawn points.' % len(spawn_points))
    
    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # TODO: uncomment below to enable autopilot
        if not self.config["scenarios"] == "straight_dynamic" and not self.config['test_comparison']:
            blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        tm_port = self.tm.get_port()
        if vehicle is not None:
            self.actor_list.append(vehicle)
            # TODO: uncomment below to enable autopilot
            if not self.config["scenarios"] == "straight_dynamic" and not self.config['test_comparison']:
                vehicle.set_autopilot(True, tm_port)

            if self.config['test_comparison']:
                self.collision_sensor_list.append(sensors.CollisionSensor(vehicle))

            if self.config["verbose"]:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False    

    def destroy_actors():
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))
