import os
import signal
import subprocess
import random


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
    
    def __del__(self):
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.server_port = None
            self.server_process = None
    
    def close(self):
        self.__del__()