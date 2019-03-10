import os
import signal
import subprocess
import random
import time

CARLA_PATH = os.environ.get("CARLA_PATH")

class CarlaServer():
    def __init__(self, config=None):
        print("Launching CARLA server...")
        self.config = config
        self.server_port = config['server_port']
        self.server_binary = config['server_binary']
        self.render_res_x = config['render_res_x']
        self.render_res_y = config['render_res_y']
        self.server_fps = config['server_fps']
        self.live_carla_processes = set()
        # TODO: Check for empty ports
        if not self.server_port:
            self.server_port = random.randint(10000, 60000)
        else:
            pass
        print(self.server_binary)
        
        launch_command = [
                self.server_binary, self.config['city_name'], "-windowed",
                "-ResX={}".format(self.render_res_x), "-ResY={}".format(self.render_res_y),
                "-carla-server", "-benchmark", "-fps={}".format(self.server_fps),
                "-carla-world-port={}".format(self.server_port), "SDL_VIDEODRIVER=offscreen",
                "SDL_HINT_CUDA_DEVICE=0"
            ]

        self.server_process = subprocess.Popen(launch_command,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"))

        if self.server_process:
            print("Launched server at port:", self.server_port)

        print('Waiting for server to finish setting up')
        time.sleep(20)
    
    def __del__(self):
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.server_port = None
            self.server_process = None
            print("Killed server process")
    
    def close(self):
        self.__del__()


if __name__ == "__main__":
    DEFAULT_ENV = {
    "server_path" : CARLA_PATH,
    "server_binary" : CARLA_PATH + '/CarlaUE4.sh',
    "server_process" : None,
    # X Rendering Resolution
    "render_res_x" : 400,
    # Y Rendering Resolution
    "render_res_y" : 300,
    # Input X Res (Default set to Atari)
    "x_res" : 84,
    # Input Y Res (Default set to Atari)
    "y_res" : 84,
    "server_fps" : 10,
    "server_port" : None,
    "city_name" : "Town01",
    "frame_skip": 1,
    "enable_planner" : True,
    "reward_function" : 'stub',
    "save_images_to_disk" : False,
    "record_sim": True,
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions" : False,
    # Number of frames stacked together
    "framestack" : 1,
    "num_vehicles" : 0,
    "num_pedestrians" : 0,
    "max_steps" : 1000,
    "next_command": None,
    "verbose": True,
    "vehicle_type": 'vehicle.toyota.prius',
    "sensors": ["r"]
    }
    server = CarlaServer(config=DEFAULT_ENV)
    time.sleep(10)
