import os
import signal
import subprocess
import random
import time
import ipdb
st = ipdb.set_trace
from environment.carla_9_4.config import DEFAULT_ENV

class CarlaServer():
    def __init__(self, config=None):
        print("Launching CARLA server...")
        self.config = config
        self.server_port = config['server_port']
        self.server_binary = config['server_binary']
        self.carla_gpu = config['carla_gpu']
        self.render_server = config['render_server']
        self.live_carla_processes = set()
        self.settings_path = os.path.join(config['server_path'],"CarlaUE4/Config/CarlaSettings.ini")
        if not self.server_port:
            self.server_port = random.randint(10000, 60000)
        else:
            pass
        print(self.server_binary)

        my_env = os.environ.copy()
        if self.carla_gpu is not None:
            my_env["SDL_HINT_CUDA_DEVICE"] = self.carla_gpu
            del my_env['CUDA_VISIBLE_DEVICES']
            print("Attempting to start carla on GPU {0}".format(self.carla_gpu))

        if not self.render_server:
            os.environ["SDL_VIDEODRIVER"] = "offscreen"
        
        '''launch_command = [
                self.server_binary, "--carla-settings={}".format(self.settings_path), "-carla-rpc-port={}".format(self.server_port)
        ]'''
        launch_command = [
                self.server_binary, "-carla-rpc-port={}".format(self.server_port)
        ]

        self.server_process = subprocess.Popen(launch_command,
            preexec_fn=os.setsid, env=my_env)

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
