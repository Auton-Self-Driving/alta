''' Planner '''
import os
import glob
import sys
import math
import numpy as np
from environment.carla_9_4.agents.navigation.global_route_planner import GlobalRoutePlanner
from environment.carla_9_4.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from collections import deque

# CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
# if CARLA_9_4_PATH == None:
#     raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

# try:
#     sys.path.append(glob.glob(CARLA_9_4_PATH+'/**/*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

try:
    import carla
except Exception as e:
    print("Failed to import Carla")
    raise e

class GlobalPlanner():

    def __init__(self):
        self._grp = None
        self._hop_resolution = 2.0
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
    
    def _trace_route(self, map, start_transform, destination_transform):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        start_waypoint, end_waypoint = map.get_waypoint(start_transform.location), map.get_waypoint(destination_transform.location)

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(map, self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route
    
    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)

    def get_next_orientation(self, vehicle_transform):
        
        next_waypoints_angles = []
        next_waypoint_found = False
        num_next_waypoints = 5
        for i, (waypoint, _) in enumerate(self._waypoints_queue):
            
            dot, angle = self.get_dot_product_and_angle(vehicle_transform, waypoint)
            
            # next_waypoint_found implies the first waypoint with 
            # positive dot product is found
            if not next_waypoint_found:
                if dot >= 0:
                    max_index = i
                    next_waypoint_found = True
                    next_waypoints_angles = [angle]
            else:
                if len(next_waypoints_angles) < num_next_waypoints:
                    next_waypoints_angles.append(angle)
                else:
                    break
        if max_index > 0:
            for i in range(max_index):
                self._waypoints_queue.popleft()
        
        if next_waypoint_found and len(next_waypoints_angles) > 0:
            angle = np.mean(np.array(next_waypoints_angles))
        else:
            angle = 0

        return angle

    def get_dot_product_and_angle(self, vehicle_transform, waypoint):

        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x -
                          v_begin.x, waypoint.transform.location.y -
                          v_begin.y, 0.0])
        dot = np.dot(w_vec, v_vec)
        angle = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            angle *= -1.0

        return dot, angle