import os
import signal
import subprocess

    
class CarlaServer():
    def __init__(self, port):
        self.port = port
        
        carla_path = os.environ.get("CARLA_8_2_PATH")
        if os.environ.get("CARLA_8_2_PATH") == None:
            raise ValueError("Set $CARLA_8_2_PATH to dir that contains CarlaUE4.sh")
            
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
