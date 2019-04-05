import carla
import random

client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.toyota.prius')

#Returns a list of carla.libcarla.Transform
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle_actor = world.spawn_actor(vehicle_bp, spawn_point)
#carla.libcarla.Transform has attributes location, rotation
camera_rgb = blueprint_library.find('sensor.camera.rgb')

# vehicle_actor.set_velocity(25)
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera_actor = world.spawn_actor(camera_rgb, camera_transform, attach_to=vehicle_actor)
vehicle_actor.set_autopilot(True)
camera_actor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))