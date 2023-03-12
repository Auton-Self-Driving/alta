import os, sys, glob
import traceback
import time

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

import carla

import environment.carla_9_4.scenarios as scenarios
import environment.carla_9_4.server as server
import environment.carla_9_4.planner as planner
import environment.carla_9_4.controller as controller
import environment.carla_9_4.sensors as sensors
from environment.carla_9_4.reward import compute_reward
from environment.carla_9_4.dashcam import Visualizer

DEFAULT_CARLA_CONFIG = {
    'server_port':None,
    'server_binary': CARLA_9_4_PATH + '/CarlaUE4.sh',
    'device':'cuda:1',
    'render_server':False,
}

def __start_carla_server(max_server_retries = 5):
    serverStarted = False
    serverStartRetries = 0
    CarlaServer = None
    while ((not serverStarted) and serverStartRetries < max_server_retries):
        try:
            CarlaServer = server.CarlaServer(config=DEFAULT_CARLA_CONFIG)
            serverStarted = True
        except Exception as e:
            print("Error in starting carla server : {}".format(traceback.format_exc()))
            CarlaServer.close()
            error = e
            serverStartRetries += 1
    return CarlaServer

def get_carla_server_and_client():

    carla_server = __start_carla_server()
    server_port_number = carla_server.server_port
    client = carla.Client('localhost', server_port_number)
    client.set_timeout(6000)

    return carla_server, client

carla_server, client = get_carla_server_and_client()

client.load_world('Town10')

world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Retrieve the spectator object
spectator = world.get_spectator()

# Set the spectator with an empty transform
# This will set the spectator at the origin of the map, with 0 degrees
# pitch, yaw and roll - a good way to orient yourself in the map
spectator.set_transform(carla.Transform())

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

new_location = carla.Transform(
                    carla.Location(location.x,location.y,250),
                    carla.Rotation(270,0,0)) # pitch(y), yaw(z), roll(x) -> left hand up z system

spectator.set_transform(new_location)

print(len(world.get_map().get_spawn_points()))

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=spectator)

# Start camera with PyGame callback
camera.listen(lambda image: image.save_to_disk('probe_out/%06d.png' % image.frame))

for i in range(3):
    world.tick()
    time.sleep(0.05)


spectator.destroy()
camera.destroy()

carla_server.close()
