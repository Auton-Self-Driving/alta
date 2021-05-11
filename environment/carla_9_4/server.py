import os
import signal
import subprocess
import random
import time
from environment.carla_9_4.config import DEFAULT_ENV

class CarlaServerWithPort:
    def __init__(self, port):
        self.port = port
        
        carla_path = os.environ.get("CARLA_9_4_PATH")
        if os.environ.get("CARLA_9_4_PATH") == None:
            raise ValueError("Set $CARLA_9_4_PATH to dir that contains CarlaUE4.sh")
            
        devnull = open(os.devnull, 'w')
        
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        
        # Clean up specified port (kill all processes)
        cmd = "kill -kill $(lsof -t -i :{})".format(port)
        subprocess.Popen(cmd, stderr=devnull, shell=True).wait()
        
        # Launch server at specified port
        cmd = ["{}CarlaUE4.sh".format(carla_path), "-carla-server",
               "-world-port={}".format(port), "-benchmark", "-fps=10", "-quality-level=Low"]  # , "-windowed", "-ResX=300", "-ResY=225"]
        self.process = subprocess.Popen(
            cmd, stderr=devnull, stdout=devnull, preexec_fn=os.setsid)
        print("[Server started] at port {}".format(port))
       
    def close(self):
        os.killpg(self.process.pid, signal.SIGKILL)
        print("[Server killed] at port {}".format(self.port))


class CarlaServer():
    def __init__(self, config=None):
        print("Launching CARLA server...")
        self.config = config
        self.server_port = config['server_port']
        self.server_binary = config['server_binary']
        self.carla_gpu = config['device'] if 'device' in config else config['carla_gpu']
        self.render_server = config['render_server']
        self.live_carla_processes = set()
        if not self.server_port:
            self.server_port = random.randint(10000, 60000)
        else:
            pass

        my_env = os.environ.copy()
        if self.carla_gpu is not None and 'cuda' in self.carla_gpu:
            my_env["SDL_HINT_CUDA_DEVICE"] = self.carla_gpu[-1:] # get GPU number
            # del my_env['CUDA_VISIBLE_DEVICES']
            print("Attempting to start carla on GPU {0}".format(self.carla_gpu))

        if not self.render_server:
            os.environ["SDL_VIDEODRIVER"] = "offscreen"

        launch_command = [
                self.server_binary, "-carla-rpc-port={}".format(self.server_port)
        ]

        self.server_process = subprocess.Popen(launch_command,
            preexec_fn=os.setsid, env=my_env)

        if self.server_process:
            print("Launched server at port [{}], pid [{}]".format(
                self.server_port, self.server_process.pid))

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
