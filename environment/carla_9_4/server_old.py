# ./CarlaUE4.sh Town01 -windowed -ResX=800 -ResY=800 -carla-server -benchmark -fps=10 -carla-world-port=2000 SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0
import os
import signal
import subprocess
import random
import time
from environment.carla_9_4.config import DEFAULT_ENV

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
        self.close()
    
    def close(self):
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.server_port = None
            self.server_process = None
            print("Killed server process")


if __name__ == "__main__":
    server = CarlaServer(config=DEFAULT_ENV)
    time.sleep(10)
