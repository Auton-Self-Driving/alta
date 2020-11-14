import sys
import os
import traceback
import time

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
ALTA = os.environ.get("ALTA")
sys.path.append(ALTA)
import environment.carla_9_4.server as server

config= {
    "server_path" : CARLA_9_4_PATH,
    "server_binary" : CARLA_9_4_PATH + '/CarlaUE4.sh',
    "server_port" : 40798,
    "carla_gpu" : "1",
    "render_server" : False
}

serverStarted = False
serverStartRetries = 0
while ((not serverStarted) and serverStartRetries < 1):
    try:
        CarlaServer = server.CarlaServer(config=config)
        serverStarted = True
    except Exception as e:
        print("Error in starting carla server : {}".format(traceback.format_exc()))
        CarlaServer.close()
        error = e
        serverStartRetries += 1
time.sleep(120)
