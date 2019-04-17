""" Scenarios file for different road paths """

import os
import glob
import sys

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+'/**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation
import random

def paths_straight_Town01():

    # paths contain list of list
    # paths = [path_1, .. , path_n]
    # path_i = [start_transform, target_transform]

    paths = [
        [
            Transform(Location(x=12.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
            Transform(Location(x=76.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
        ],
        [
            Transform(Location(x=92.11, y=316, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0)),
            Transform(Location(x=92.11, y=213, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0))
        ],
        [
            Transform(Location(x=324.0, y=129.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0)),
            Transform(Location(x=108.5, y=199.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0))
        ],
        [
            Transform(Location(x=102.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0)),
            Transform(Location(x=320.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0))
        ],
        [
            Transform(Location(x=140.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
            Transform(Location(x=320.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
        ]
    ]

    return paths

def get_fixed_short_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return paths_straight_Town01()[0]

def get_fixed_long_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return paths_straight_Town01()[1]

def get_random_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return random.choice(paths_straight_Town01())
