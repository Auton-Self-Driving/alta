import numpy as np
import math
import os
import sys
import pyproj
import glob

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(".egg file not found! Kindly check for your Carla installation.")
    pass

import carla
from carla.libcarla import Location

def _create_bb_points(vehicle):
        """
        Returns lower plane of 3D bounding box for a vehicle.
        """
        cords = np.zeros((4, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        return cords

def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = get_matrix(bb_transform)
        vehicle_world_matrix = get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

def get_bounding_box(vehicle):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = _create_bb_points(vehicle)
        cords_x_y_z = _vehicle_to_world(bb_cords, vehicle)[:3, :]
        return np.array(cords_x_y_z)

def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

def get_wp_from_bb(bbox_cords, world_map):
    """
    Get list of wp from given bbox_cords array
    """

    dimension, num_pts = bbox_cords.shape
    assert(dimension == 3)

    bb_wps = []
    for i in range(num_pts):
        vertex = bbox_cords[:,i]
        vertex_location = Location(x=vertex[0], y=vertex[1], z=vertex[2])
        vertex_wp = world_map.get_waypoint(vertex_location)
        bb_wps.append(vertex_wp)

    return bb_wps

def get_road_lane_id_set_from_wp(wp_list):

    road_lane_id_set = set()
    for wp in wp_list:
        road_lane_id_set.add((wp.road_id, wp.lane_id))

    return road_lane_id_set

def get_vehicle_bb_wp(world_map, vehicle):
    '''
    Get waypoints corresponding to vehicle centre and vertices of bb of vehicle
    '''
    vehicle_wp = world_map.get_waypoint(vehicle.get_location())
    vehicle_bb = get_bounding_box(vehicle)

    vehicle_bb_wp = get_wp_from_bb(vehicle_bb, world_map)
    vehicle_bb_wp.append(vehicle_wp)

    return vehicle_bb_wp

def check_if_vehicle_in_same_lane(vehicle_actor, target_vehicle, next_waypoints, world_map):
    '''
    Checks if target_vehicle is in same lane/road as vehicle_actor or next_waypoints
    '''

    target_vehicle_bb_wp = get_vehicle_bb_wp(world_map, target_vehicle)
    target_vehicle_road_id_set = get_road_lane_id_set_from_wp(target_vehicle_bb_wp)

    vehicle_bb_wp = get_vehicle_bb_wp(world_map, vehicle_actor)
    vehicle_wp_list = vehicle_bb_wp + next_waypoints
    vehicle_road_id_set = get_road_lane_id_set_from_wp(vehicle_wp_list)

    intersection_set = target_vehicle_road_id_set & vehicle_road_id_set

    return (len(intersection_set) > 0)

def _latlon_to_ecef(lat,lon,alt):
    # Projections
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Transform from lat/lon to ecef
    x,y,z= pyproj.transform(p1=lla,
        p2 = ecef,
        x = lon,
        y = lat,
        z = alt,
        radians=False)

    return x, y, z

def _ecef_to_latlon(x,y,z):
    # Projections
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Transform from lat/lon to ecef
    lon, lat, alt= pyproj.transform(p1=ecef,
        p2 = lla,
        x = x,
        y = y,
        z = z,
        radians=False)

    return lat, lon, alt

def get_world_coords_from_latlong(latitude, longitude, altitude, world_map):
    origin_latlong = world_map.transform_to_geolocation(carla.Location())

    # Origin in ECEF coordinates
    O_ecef = _latlon_to_ecef(origin_latlong.latitude, origin_latlong.longitude, origin_latlong.altitude)

    # Convert GNSS data to ECEF coordinates
    P_ecef = _latlon_to_ecef(latitude, longitude, altitude)

    # Calculate difference between current location and origin
    #FIXME The /2 constant is a hacky fix to get this working - this shouldn't be here
    delta = np.expand_dims(np.array(P_ecef) - np.array(O_ecef), axis = 1)

    # Create the rotation matrix to convert from ECEF to ENU Coords
    ecef_to_enu_rot = np.array(
        [[-np.sin(longitude), np.cos(longitude), 0],
         [-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)],
         [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]]
    )
    enu = ecef_to_enu_rot@delta

    # Create rotation matrix to convert from right hand ENU frame to left-hand CARLA frame
    enu_to_carla_rot = np.array(
        [[1, 0, 0],
         [0,-1, 0],
         [0, 0, 1]]
    )

    return enu_to_carla_rot@enu

def convert_route_from_GPS_world(route, world_map):

    # Example route input
    # route =[({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002271601998707}, RoadOption.LEFT),
    #     ({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002709765148996}, RoadOption.RIGHT),
    #     ({'z': 0.0, 'lat': 48.99822679980298, 'lon': 8.002735250105061}, RoadOption.STRAIGHT)
    #     ]

    mapped_route = []
    for idx, pt in enumerate(route):
        print(pt)
        altitude = pt[0]['z']
        latitude = pt[0]['lat']
        longitude = pt[0]['lon']
        world_coord = get_world_coords_from_latlong(latitude, longitude, altitude, world_map)
        x, y, z = world_coord[0][0], world_coord[1][0], world_coord[2][0]
        mapped_route.append(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation()))
    return mapped_route


########## Velocity related utilities #########

def get_speed_from_velocity(velocity):
    """ Converts velocity to speed
    """
    speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
    return speed

def cosine_between_velocities(v1, v2): 
    """ Computes the cosine of the angle between two velocity vectors
    """
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / \
        (get_speed_from_velocity(v1) * get_speed_from_velocity(v2) + 1e-6)

def cosine_between_obs(agt_v, obs_v, zero_speed_threshold = 0.05):
    """ Computes the cosine of the angle between velocity vectors of vehicles
    where at least one of them is in motion.
    """
    # Return 0 if both vehicles are stationary
    # TODO: Check - doesn't this mean they are perpendicular?
    if get_speed_from_velocity(agt_v) < zero_speed_threshold or \
        get_speed_from_velocity(obs_v) < zero_speed_threshold:
        return 0.
    else:
        return cosine_between_velocities(agt_v, obs_v)


######## Agent State Related Utilities #########

def fetch_actor_features(actor):
    """ Retrieves a brief set of relevant features of an actor
    """
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

    bounding_box_loc = actor.bounding_box.get_world_vertices(transform)
    bounding_box = [(loc.x, loc.y) for loc in bounding_box_loc]

    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'theta': transform.rotation.yaw,
        'speed': speed,
        'bounding_box': bounding_box
    }

def normalize_actor_features(actor_features, ref, theta):
        """
        Normalize actor feature dictionary to reference point
        ref is a tuple (x, y, theta)
        """
        for i, (x,y) in enumerate(actor_features['bounding_box']):
            x,y = transform_to_pov((x,y), ref, theta)
            actor_features['bounding_box'][i] = (x,y)

        x,y = transform_to_pov((actor_features['x'], actor_features['y']), ref, theta)
        actor_features['x'], actor_features['y'] = x,y
        actor_features['theta'] = normalize_angle(actor_features['theta'] - theta)

def _is_static(agent, zero_speed_threshold):
        if type(agent.obstacle_sensor) == dict:
            for suffix in agent.obstacle_sensor:
                obstacle_key = 'obstacle_dist_{}'.format(suffix)
                if obstacle_key in agent.episode_measurements and \
                    agent.episode_measurements[obstacle_key] != -1:
                    return False
        if agent.episode_measurements['speed'] >= zero_speed_threshold:
            return False
        if agent.episode_measurements['obstacle_dist'] != -1:
            return False
        if agent.episode_measurements['red_light_dist'] != -1:
            return False
        if  'obstacle_dist_left' in agent.episode_measurements and \
            agent.episode_measurements['obstacle_dist_left'] != -1:
            return False
        if  'obstacle_dist_right' in agent.episode_measurements and \
            agent.episode_measurements['obstacle_dist_right'] != -1:
            return False
        return True

def get_wp_obs_input(agent):
        '''
        Create wp angles input as list
        TODO: Consider moving to a file related to observation spaces
        '''
        num_wp = 5
        wp_angles_array = None
        wp_vectors_array = None

        n = len(agent.next_wp_angles)
        if n == 0:
            print("No next waypoints found. Giving zero as input.")
            wp_angles_array = np.zeros(num_wp)
            wp_vectors_array = np.zeros(2 * num_wp)

        elif n == num_wp:
            wp_angles_array = np.array(agent.next_wp_angles)
            wp_vectors_array = np.array(agent.next_wp_vectors)

        elif n < num_wp:
            # Fill using last entry
            last_angle = agent.next_wp_angles[-1]
            last_vec = agent.next_wp_vectors[-1]

            for _ in range(num_wp-n):
                agent.next_wp_angles.append(last_angle)
                agent.next_wp_vectors.append(last_vec)
            wp_angles_array = np.array(agent.next_wp_angles)
            wp_vectors_array = np.array(agent.next_wp_vectors)
        else:
            print("Error: More than {0} waypoints returned from planner.".format(num_wp))
            print("Taking required number of entries.")
            wp_angles_array = np.array(agent.next_wp_angles[:num_wp])
            wp_vectors_array = np.array(agent.next_wp_vectors[:num_wp])

        wp_vectors_array = wp_vectors_array.reshape(-1)

        return wp_angles_array, wp_vectors_array

def is_within_distance_ahead(target_transform, current_transform, max_distance):
        """
        Check if a target object is within a certain distance in front of a reference object.
        :param target_transform: location of the target object
        :param current_transform: location of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
        """
        target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
        norm_target = np.linalg.norm(target_vector)

        # If the vector is too short, we can simply stop here
        if norm_target < 0.001:
            return True, 0, norm_target

        if norm_target > max_distance:
            return False, -1, norm_target

        fwd = current_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return d_angle < 90.0, d_angle, norm_target


####### Temporal Action Space related utilities ########

class CubicBezierHandler:

    def __init__(self, points_on_traj):
        """
        Args:
            points_on_traj (int): The number of points (T) to represent the bezier curve shape
        Instance Attributes:
            time_slices (np.ndarray): A (T,) numpy array with T entries in [0,1] having step size 1/T
            cubic_bezier_matrix (np.ndarray): A (T,4) numpy array of cubic bezier coefficients having 1/T time resolution
        """

        self.points_on_traj = points_on_traj
        self.time_slices = np.linspace(0,1,points_on_traj)
        self.cubic_bezier_matrix = np.asarray([
            (1-self.time_slices)**3, 
            (3*self.time_slices*(1-self.time_slices)**2), 
            3*(1-self.time_slices)*self.time_slices**2, 
            self.time_slices**3
        ]).T
        self.ctrl_points = None

    def set_ctrl_points(self, ctrl_points, convert_to_np = True):
        """Stores ctrl points that parameterize a cubic bezier curve 

        Args:
            ctrl_points ([List,np.ndarray]): 4x2 array or (8,) list of ctrl points describing a cubic bezier curve
            convert_to_np (bool): Set to True if passing ctrl_points as list
        """

        if convert_to_np:
            ctrl_points = np.asarray(ctrl_points)
            ctrl_points = ctrl_points.reshape(4,2)

        self.ctrl_points = ctrl_points

    def __get_cubic_bezier_points(self):
        """Returns points on a cubic bezier parameterized by ctrl_points. 

        Returns:
            np.ndarray: Tx2 array of points on the specified bezier curve. 
        """

        pts_on_curve = self.cubic_bezier_matrix @ self.ctrl_points

        return pts_on_curve

    def get_sub_cubic_bezier_ctrl_points(self, start_time, free_pts):
        """Computes new ctrl points to represent a segment of an existing cubic bezier curve from a specified point
        until the end point.

        Args:
            start_time (float): Determines the point on the original bezier curve to use a the new start point. In [0,1]

        Returns:
            np.ndarray: 4x2 array of control points describing the specified segment of the cubic bezier centred at the origin.

        Notes:
            orig_pts (np.ndarray): Tx2 array of points along orig bezier curve, where 1/T gives the timeslice each entry represents 
        """

        query_idx = lambda idx : int(idx*self.points_on_traj) # idx is a float in [0,1]
        orig_pts = self.__get_cubic_bezier_points()

        start_pt = orig_pts[query_idx(start_time),:] 

        t_n = (self.time_slices[query_idx(start_time):] - start_time) / (1 - start_time)
        T = np.asarray([(1-t_n)**3, (3*t_n*(1-t_n)**2), 3*(1-t_n)*t_n**2, t_n**3]).T
        Y = np.copy(orig_pts[query_idx(start_time):,:])

        new_ctrl_pts = np.copy(self.ctrl_points)        
        new_ctrl_pts[0,:] = start_pt

        # 1 free pt
        if free_pts == 1:

            smoothing_factor = 2.5
            new_ctrl_pts[1,0] = ((self.ctrl_points[3,0] - orig_pts[query_idx(start_time),0] )/ smoothing_factor) + orig_pts[query_idx(start_time),0]
            new_ctrl_pts[1,1] = orig_pts[query_idx(start_time),1]

            Y -= T[:,(0,1,3)] @ new_ctrl_pts[(0,1,3),:]
            new_ctrl_pts[2:3,:] = np.linalg.inv(T[:,2:3].T @ T[:,2:3]) @ T[:,2:3].T @ Y 

        # 2 free pts
        elif free_pts == 2:
            Y -= T[:,(0,3)] @ new_ctrl_pts[(0,3),:]
            new_ctrl_pts[1:3,:] = np.linalg.inv(T[:,1:3].T @ T[:,1:3]) @ T[:,1:3].T @ Y 

        # Centre coordinates
        new_ctrl_pts -= new_ctrl_pts[0,:]

        return new_ctrl_pts


def bezier_to_action(ctrl_points, pt_range, pt2ac, free_pts):

    action = []

    if free_pts == 2:
        action.append(pt2ac(pt_range[1]['x'], ctrl_points[1,0]))
        action.append(pt2ac(pt_range[1]['y'], ctrl_points[1,1]))
        action.append(pt2ac(pt_range[2]['x'], ctrl_points[2,0]))
        action.append(pt2ac(pt_range[2]['y'], ctrl_points[2,1]))
        action.append(pt2ac(pt_range[3]['y'], ctrl_points[3,1]))

    return action


######## STATIC MAPPINGS ###################


SEMANTIC_COLOR_MAP = {
    0	: ["Unlabeled", ( 0, 0, 0)],
    1	: ["Building",	( 70, 70, 70)],
    2	: ["Fence",	(190, 153, 153)],
    3	: ["Other",	(250, 170, 160)],
    4	: ["Pedestrian",	(220, 20, 60)],
    5	: ["Pole",	(153, 153, 153)],
    6	: ["Road line",	(157, 234, 50)],
    7	: ["Road",	(128, 64, 128)],
    8	: ["Sidewalk",	(244, 35, 232)],
    9	: ["Vegetation",	(107, 142, 35)],
    10	: ["Car",	( 0, 0, 142)],
    11	: ["Wall",	(102, 102, 156)],
    12	: ["Traffic sign",	(220, 220, 0)]
}

SEMANTIC_COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],
    [70, 70, 70],
    [190, 153, 153],
    [250, 170, 160],
    [220, 20, 60],
    [153, 153, 153],
    [157, 234, 50],
    [128, 64, 128],
    [244, 35, 232],
    [107, 142, 35],
    [0, 0, 142],
    [102, 102, 156],
    [220, 220, 0]
])

CLASS_REMAP = {
    0	: 0,
    1	: 0,
    2	: 0,
    3	: 0,
    4	: 1,
    5	: 0,
    6	: 2,
    7	: 3,
    8	: 0,
    9	: 0,
    10	: 4,
    11	: 0,
    12	: 0,
    13  : 0,
    14  : 0,
    15  : 0,
    16  : 0,
    17  : 0,
    18  : 0,
    19  : 0,
    20  : 0,
    21  : 0,
    22  : 0
}

CLASS_REMAP_ARRAY = np.array([
    0,
    0,
    0,
    0,
    1,
    0,
    2,
    3,
    0,
    0,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
])

BINARIZED_REMAP_ARRAY = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
])

REDUCED_SEMANTIC_COLOR_MAP = {
    0	: ["Everything Else", ( 0, 0, 0)],
    1	: ["Pedestrian",	(220, 20, 60)],
    2	: ["Road line",	(157, 234, 50)],
    3	: ["Road",	(128, 64, 128)],
    4	: ["Car",	( 0, 0, 142)]
}

REDUCED_SEMANTIC_COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],
    [220, 20, 60],
    [157, 234, 50],
    [128, 64, 128],
    [0, 0, 142]
])

BINARIZED_SEMANTIC_COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],
    [255, 255, 255]
])

def reduce_classes(semantic_image, binarized_image=False):
    h, w = np.shape(semantic_image)
    # # assert(d == 1)
    # semantic_reduced_image = np.zeros_like(semantic_image)
    if binarized_image:
        f = lambda x : BINARIZED_REMAP_ARRAY[x]
    else:
        f = lambda x : CLASS_REMAP_ARRAY[x]
    # print(semantic_image.reshape(-1))
    semantic_reduced_image = f(semantic_image.reshape(-1))
    return semantic_reduced_image.reshape((h,w))


def convert_to_one_hot(labels, num_classes):
    labels = np.squeeze(labels)
    h, w = labels.shape
    flattened_labels = labels.reshape((h*w))
    one_hot = np.zeros((flattened_labels.shape[0], num_classes))
    one_hot[np.arange(flattened_labels.shape[0]), flattened_labels] = 1
    one_hot = one_hot.reshape((h, w, -1))

    return one_hot

def convert_from_one_hot(one_hot):
    return np.argmax(one_hot, axis=2)


def convert_to_rgb(semantic_image, reduced_classes=False, binarized_image=False):
    h, w = np.shape(semantic_image)
    semantic_rgb_image = np.zeros((h, w, 3))

    if reduced_classes:
        if binarized_image:
            semantic_map = BINARIZED_SEMANTIC_COLOR_MAP_ARRAY
        else:
            semantic_map = REDUCED_SEMANTIC_COLOR_MAP_ARRAY
    else:
        semantic_map = SEMANTIC_COLOR_MAP_ARRAY

    f = lambda x : semantic_map[x]

    semantic_rgb_image = f(semantic_image.reshape(-1))
    return semantic_rgb_image.reshape((h,w,3))

