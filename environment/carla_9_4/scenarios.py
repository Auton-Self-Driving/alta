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

WAYPOINT_DICT = {
    0: Transform(Location(x=271.0400085449219, y=129.489990234375, z=1.32), Rotation(yaw=179.999755859375)),
    1: Transform(Location(x=270.79998779296875, y=133.43003845214844, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    2: Transform(Location(x=237.6999969482422, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    3: Transform(Location(x=237.6999969482422, y=133.239990234375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    4: Transform(Location(x=216.26998901367188, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    5: Transform(Location(x=216.26998901367188, y=133.239990234375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    6: Transform(Location(x=191.3199920654297, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    7: Transform(Location(x=191.3199920654297, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    8: Transform(Location(x=157.1899871826172, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    9: Transform(Location(x=157.1899871826172, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    10: Transform(Location(x=338.97998046875, y=301.2599792480469, z=1.32), Rotation(yaw=-90.00029754638672)),
    11: Transform(Location(x=128.94998168945312, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    12: Transform(Location(x=128.94998168945312, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    13: Transform(Location(x=119.46998596191406, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    14: Transform(Location(x=105.43998718261719, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    15: Transform(Location(x=92.11000061035156, y=39.709999084472656, z=1.32), Rotation(yaw=-90.00029754638672)),
    16: Transform(Location(x=88.6199951171875, y=26.559999465942383, z=1.32), Rotation(yaw=90.00004577636719)),
    17: Transform(Location(x=92.11000061035156, y=30.820009231567383, z=1.32), Rotation(yaw=-90.00029754638672)),
    18: Transform(Location(x=88.6199951171875, y=15.279999732971191, z=1.32), Rotation(yaw=90.00004577636719)),
    19: Transform(Location(x=92.11000061035156, y=86.95999908447266, z=1.32), Rotation(yaw=-90.00029754638672)),
    20: Transform(Location(x=88.6199951171875, y=72.6199951171875, z=1.32), Rotation(yaw=90.00004577636719)),
    21: Transform(Location(x=335.489990234375, y=298.80999755859375, z=1.32), Rotation(yaw=90.00004577636719)),
    22: Transform(Location(x=92.1099853515625, y=95.44999694824219, z=1.32), Rotation(yaw=-90.00029754638672)),
    23: Transform(Location(x=88.61998748779297, y=95.44999694824219, z=1.32), Rotation(yaw=90.00004577636719)),
    24: Transform(Location(x=92.1099853515625, y=113.05999755859375, z=1.32), Rotation(yaw=-90.00029754638672)),
    25: Transform(Location(x=88.61998748779297, y=103.37999725341797, z=1.32), Rotation(yaw=90.00004577636719)),
    26: Transform(Location(x=92.1099853515625, y=159.9499969482422, z=1.32), Rotation(yaw=-90.00029754638672)),
    27: Transform(Location(x=88.61998748779297, y=145.83999633789062, z=1.32), Rotation(yaw=90.00004577636719)),
    28: Transform(Location(x=92.1099853515625, y=176.88999938964844, z=1.32), Rotation(yaw=-90.00029754638672)),
    29: Transform(Location(x=88.61998748779297, y=169.84999084472656, z=1.32), Rotation(yaw=90.00004577636719)),
    30: Transform(Location(x=-2.4200193881988525, y=187.97000122070312, z=1.32), Rotation(yaw=89.9996109008789)),
    31: Transform(Location(x=1.5599803924560547, y=187.9700164794922, z=1.32), Rotation(yaw=-90.00040435791016)),
    32: Transform(Location(x=338.97998046875, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    33: Transform(Location(x=-2.4200096130371094, y=149.8300018310547, z=1.32), Rotation(yaw=89.9996109008789)),
    34: Transform(Location(x=1.5599901676177979, y=149.83001708984375, z=1.32), Rotation(yaw=-90.00040435791016)),
    35: Transform(Location(x=-2.4200096130371094, y=120.0199966430664, z=1.32), Rotation(yaw=89.9996109008789)),
    36: Transform(Location(x=1.5599901676177979, y=120.02001953125, z=1.32), Rotation(yaw=-90.00040435791016)),
    37: Transform(Location(x=-2.4200048446655273, y=79.31999969482422, z=1.32), Rotation(yaw=89.9996109008789)),
    38: Transform(Location(x=1.5599950551986694, y=79.32001495361328, z=1.32), Rotation(yaw=-90.00040435791016)),
    39: Transform(Location(x=-2.4200048446655273, y=48.70000076293945, z=1.32), Rotation(yaw=89.9996109008789)),
    40: Transform(Location(x=1.5599950551986694, y=48.70001983642578, z=1.32), Rotation(yaw=-90.00040435791016)),
    41: Transform(Location(x=-2.420001268386841, y=17.779998779296875, z=1.32), Rotation(yaw=89.9996109008789)),
    42: Transform(Location(x=1.55999755859375, y=22.440019607543945, z=1.32), Rotation(yaw=-90.00040435791016)),
    43: Transform(Location(x=335.489990234375, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    44: Transform(Location(x=21.770000457763672, y=-1.9599987268447876, z=1.32), Rotation(yaw=179.9996337890625)),
    45: Transform(Location(x=14.139999389648438, y=2.0200109481811523, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    46: Transform(Location(x=47.939998626708984, y=-1.9599950313568115, z=1.32), Rotation(yaw=179.9996337890625)),
    47: Transform(Location(x=47.939998626708984, y=2.020014524459839, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    48: Transform(Location(x=72.5999984741211, y=-1.9599950313568115, z=1.32), Rotation(yaw=179.9996337890625)),
    49: Transform(Location(x=62.12999725341797, y=2.020014524459839, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    50: Transform(Location(x=116.63999938964844, y=-1.95999014377594, z=1.32), Rotation(yaw=179.9996337890625)),
    51: Transform(Location(x=110.02999877929688, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    52: Transform(Location(x=137.7899932861328, y=-1.95999014377594, z=1.32), Rotation(yaw=179.9996337890625)),
    53: Transform(Location(x=126.38999938964844, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    54: Transform(Location(x=338.97998046875, y=226.75, z=1.32), Rotation(yaw=-90.00029754638672)),
    55: Transform(Location(x=185.55999755859375, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    56: Transform(Location(x=173.14999389648438, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    57: Transform(Location(x=209.5800018310547, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    58: Transform(Location(x=209.5800018310547, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    59: Transform(Location(x=244.09999084472656, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    60: Transform(Location(x=244.09999084472656, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    61: Transform(Location(x=278.80999755859375, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    62: Transform(Location(x=278.80999755859375, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    63: Transform(Location(x=316.8500061035156, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    64: Transform(Location(x=306.28997802734375, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    65: Transform(Location(x=334.8299865722656, y=217.0800018310547, z=1.32), Rotation(yaw=90.00004577636719)),
    66: Transform(Location(x=363.0, y=-1.9599609375, z=1.32), Rotation(yaw=179.9996337890625)),
    67: Transform(Location(x=356.79998779296875, y=2.0200390815734863, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    68: Transform(Location(x=378.17999267578125, y=-1.9599609375, z=1.32), Rotation(yaw=179.9996337890625)),
    69: Transform(Location(x=378.17999267578125, y=2.0200390815734863, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    70: Transform(Location(x=396.4499816894531, y=19.9200382232666, z=1.32), Rotation(yaw=-90.00029754638672)),
    71: Transform(Location(x=392.4700012207031, y=19.9200382232666, z=1.32), Rotation(yaw=90.00004577636719)),
    72: Transform(Location(x=395.9599914550781, y=164.1699981689453, z=1.32), Rotation(yaw=-90.00029754638672)),
    73: Transform(Location(x=392.4700012207031, y=164.1699981689453, z=1.32), Rotation(yaw=90.00004577636719)),
    74: Transform(Location(x=395.9599914550781, y=105.38999938964844, z=1.32), Rotation(yaw=-90.00029754638672)),
    75: Transform(Location(x=392.4700012207031, y=105.38999938964844, z=1.32), Rotation(yaw=90.00004577636719)),
    76: Transform(Location(x=395.9599914550781, y=68.86003875732422, z=1.32), Rotation(yaw=-90.00029754638672)),
    77: Transform(Location(x=392.4700012207031, y=68.86003875732422, z=1.32), Rotation(yaw=90.00004577636719)),
    78: Transform(Location(x=395.9599914550781, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    79: Transform(Location(x=392.4700012207031, y=308.2099914550781, z=1.32), Rotation(yaw=90.00004577636719)),
    80: Transform(Location(x=395.9599914550781, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    81: Transform(Location(x=392.4700012207031, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    82: Transform(Location(x=395.9599914550781, y=212.89999389648438, z=1.32), Rotation(yaw=-90.00029754638672)),
    83: Transform(Location(x=392.4700012207031, y=212.89999389648438, z=1.32), Rotation(yaw=90.00004577636719)),
    84: Transform(Location(x=1.5099804401397705, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    85: Transform(Location(x=-1.2800195217132568, y=309.4599914550781, z=1.32), Rotation(yaw=90.00004577636719)),
    86: Transform(Location(x=1.5099804401397705, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    87: Transform(Location(x=-1.980019450187683, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    88: Transform(Location(x=121.22996520996094, y=195.00999450683594, z=1.32), Rotation(yaw=179.999755859375)),
    89: Transform(Location(x=105.22998809814453, y=198.5, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    90: Transform(Location(x=118.94999694824219, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    91: Transform(Location(x=111.56999969482422, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    92: Transform(Location(x=141.12998962402344, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    93: Transform(Location(x=125.9699935913086, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    94: Transform(Location(x=22.17997932434082, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    95: Transform(Location(x=22.17997932434082, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    96: Transform(Location(x=92.10997772216797, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    97: Transform(Location(x=46.14997863769531, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    98: Transform(Location(x=46.14997863769531, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    99: Transform(Location(x=65.3499755859375, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    100: Transform(Location(x=60.10997772216797, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    101: Transform(Location(x=381.3399963378906, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    102: Transform(Location(x=381.3399658203125, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    103: Transform(Location(x=366.53997802734375, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    104: Transform(Location(x=358.39996337890625, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    105: Transform(Location(x=320.8699645996094, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    106: Transform(Location(x=306.76995849609375, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    107: Transform(Location(x=88.61997985839844, y=295.32000732421875, z=1.32), Rotation(yaw=90.00004577636719)),
    108: Transform(Location(x=301.3399658203125, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    109: Transform(Location(x=301.3399658203125, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    110: Transform(Location(x=262.5999755859375, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    111: Transform(Location(x=262.5999755859375, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    112: Transform(Location(x=232.19998168945312, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    113: Transform(Location(x=232.19998168945312, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    114: Transform(Location(x=199.94998168945312, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    115: Transform(Location(x=199.94998168945312, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    116: Transform(Location(x=173.11997985839844, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    117: Transform(Location(x=173.11997985839844, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    118: Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    119: Transform(Location(x=124.73997497558594, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    120: Transform(Location(x=114.3499755859375, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    121: Transform(Location(x=142.91998291015625, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    122: Transform(Location(x=142.91998291015625, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    123: Transform(Location(x=142.91998291015625, y=195.26998901367188, z=1.32), Rotation(yaw=179.999755859375)),
    124: Transform(Location(x=142.91998291015625, y=198.75999450683594, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    125: Transform(Location(x=178.7699737548828, y=195.26998901367188, z=1.32), Rotation(yaw=179.999755859375)),
    126: Transform(Location(x=178.7699737548828, y=198.75999450683594, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    127: Transform(Location(x=217.50997924804688, y=195.26998901367188, z=1.32), Rotation(yaw=179.999755859375)),
    128: Transform(Location(x=217.50997924804688, y=198.75999450683594, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    129: Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    130: Transform(Location(x=256.3499755859375, y=195.5699920654297, z=1.32), Rotation(yaw=179.999755859375)),
    131: Transform(Location(x=256.3499755859375, y=199.05999755859375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    132: Transform(Location(x=299.39996337890625, y=195.5699920654297, z=1.32), Rotation(yaw=179.999755859375)),
    133: Transform(Location(x=299.39996337890625, y=199.05999755859375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    134: Transform(Location(x=158.0800018310547, y=27.18000030517578, z=1.32), Rotation(yaw=-90.00029754638672)),
    135: Transform(Location(x=153.75999450683594, y=18.889999389648438, z=1.32), Rotation(yaw=90.00004577636719)),
    136: Transform(Location(x=157.25, y=39.709999084472656, z=1.32), Rotation(yaw=-90.00029754638672)),
    137: Transform(Location(x=153.75999450683594, y=28.899999618530273, z=1.32), Rotation(yaw=90.00004577636719)),
    138: Transform(Location(x=191.0800018310547, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    139: Transform(Location(x=172.2899932861328, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    140: Transform(Location(x=92.1099853515625, y=227.22000122070312, z=1.32), Rotation(yaw=-90.00029754638672)),
    141: Transform(Location(x=202.5500030517578, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    142: Transform(Location(x=202.5500030517578, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    143: Transform(Location(x=234.26998901367188, y=55.84001922607422, z=1.32), Rotation(yaw=179.999755859375)),
    144: Transform(Location(x=234.26998901367188, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    145: Transform(Location(x=272.2900085449219, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    146: Transform(Location(x=272.2900085449219, y=59.33003616333008, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    147: Transform(Location(x=299.3999938964844, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    148: Transform(Location(x=299.3999938964844, y=59.33003616333008, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    149: Transform(Location(x=299.3999938964844, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    150: Transform(Location(x=299.3999938964844, y=133.2400360107422, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    151: Transform(Location(x=88.61998748779297, y=212.89999389648438, z=1.32), Rotation(yaw=90.00004577636719))
}


def paths_straight_Town01_train():

    # paths contain list of list
    # paths = [path_1, .. , path_n]
    # path_i = [start_transform, target_transform]

    paths = [
       [
           WAYPOINT_DICT[36],
           WAYPOINT_DICT[40]
       ],
       [
           WAYPOINT_DICT[7],
           WAYPOINT_DICT[3]
       ],
       [
           WAYPOINT_DICT[110],
           WAYPOINT_DICT[114]
       ],
       [
           WAYPOINT_DICT[68],
           WAYPOINT_DICT[50]
       ],
       [
           WAYPOINT_DICT[147],
           WAYPOINT_DICT[90]
       ],
       [
           WAYPOINT_DICT[33],
           WAYPOINT_DICT[87]
       ],
       [
           WAYPOINT_DICT[80],
           WAYPOINT_DICT[76]
       ],
       [
           WAYPOINT_DICT[45],
           WAYPOINT_DICT[49]
       ],
       [
           WAYPOINT_DICT[95],
           WAYPOINT_DICT[104]
       ],
       [
           WAYPOINT_DICT[20],
           WAYPOINT_DICT[107]
       ],
       [
           WAYPOINT_DICT[78],
           WAYPOINT_DICT[70]
       ],
       [
           WAYPOINT_DICT[68],
           WAYPOINT_DICT[44]
       ],
       [
           WAYPOINT_DICT[45],
           WAYPOINT_DICT[69]
       ]
    ]
    # paths = [
    #    [
    #        WAYPOINT_DICT[0],
    #        WAYPOINT_DICT[8]
    #    ],
    #    [
    #        WAYPOINT_DICT[130],
    #        WAYPOINT_DICT[123]
    #    ]
    # ]
    # paths = [
    #     [
    #         Transform(Location(x=12.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
    #         Transform(Location(x=76.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=92.11, y=316, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0)),
    #         Transform(Location(x=92.11, y=213, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=324.0, y=129.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0)),
    #         Transform(Location(x=108.5, y=199.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=102.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0)),
    #         Transform(Location(x=320.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=140.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
    #         Transform(Location(x=320.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
    #     ]
    # ]

    return paths

def paths_straight_Town01_test():
    paths = [
       [
           WAYPOINT_DICT[39],
           WAYPOINT_DICT[35]
       ],
       [
           WAYPOINT_DICT[0],
           WAYPOINT_DICT[4]
       ],
       [
           WAYPOINT_DICT[61],
           WAYPOINT_DICT[59]
       ],
       [
           WAYPOINT_DICT[55],
           WAYPOINT_DICT[44]
       ],
       [
           WAYPOINT_DICT[47],
           WAYPOINT_DICT[64]
       ],
       [
           WAYPOINT_DICT[26],
           WAYPOINT_DICT[19]
       ],
       [
           WAYPOINT_DICT[29],
           WAYPOINT_DICT[107]
       ],
       [
           WAYPOINT_DICT[84],
           WAYPOINT_DICT[34]
       ],
       [
           WAYPOINT_DICT[53],
           WAYPOINT_DICT[67]
       ],
       [
           WAYPOINT_DICT[22],
           WAYPOINT_DICT[17]
       ],
       [
           WAYPOINT_DICT[91],
           WAYPOINT_DICT[148]
       ],
       [
           WAYPOINT_DICT[95],
           WAYPOINT_DICT[102]
       ]
    ]
    
    return paths

def paths_left_Town01_train():
    
    paths = [
        [
            WAYPOINT_DICT[85],
            WAYPOINT_DICT[98]
        ],
        [
            WAYPOINT_DICT[87],
            WAYPOINT_DICT[100]
        ],
        [
            WAYPOINT_DICT[85],
            WAYPOINT_DICT[100]
        ],
        [
            WAYPOINT_DICT[87],
            WAYPOINT_DICT[98]
        ],
        [
            WAYPOINT_DICT[46],
            WAYPOINT_DICT[37]
        ],
        [
            WAYPOINT_DICT[44],
            WAYPOINT_DICT[39]
        ],
        [
            WAYPOINT_DICT[48],
            WAYPOINT_DICT[37]
        ],
        [
            WAYPOINT_DICT[46],
            WAYPOINT_DICT[39]
        ],
        [
            WAYPOINT_DICT[44],
            WAYPOINT_DICT[37]
        ],
        [
            WAYPOINT_DICT[48],
            WAYPOINT_DICT[39]
        ]
    ]
    return paths

def paths_left_Town01_test():
    
    paths = [
        [
            WAYPOINT_DICT[76],
            WAYPOINT_DICT[63]
        ],
        [
            WAYPOINT_DICT[70],
            WAYPOINT_DICT[66]
        ],
        [
            WAYPOINT_DICT[76],
            WAYPOINT_DICT[77]
        ],
        [
            WAYPOINT_DICT[70],
            WAYPOINT_DICT[63]
        ],
        [
            WAYPOINT_DICT[104],
            WAYPOINT_DICT[78]
        ],
        [
            WAYPOINT_DICT[106],
            WAYPOINT_DICT[80]
        ],
        [
            WAYPOINT_DICT[104],
            WAYPOINT_DICT[80]
        ],
        [
            WAYPOINT_DICT[106],
            WAYPOINT_DICT[78]
        ]
    ]
    return paths

def paths_right_Town01_train():
    
    paths = [
        [
            WAYPOINT_DICT[42],
            WAYPOINT_DICT[49]
        ],
        [
            WAYPOINT_DICT[40],
            WAYPOINT_DICT[47]
        ],
        [
            WAYPOINT_DICT[42],
            WAYPOINT_DICT[47]
        ],
        [
            WAYPOINT_DICT[40],
            WAYPOINT_DICT[49]
        ],
        [
            WAYPOINT_DICT[38],
            WAYPOINT_DICT[49]
        ],
        [
            WAYPOINT_DICT[38],
            WAYPOINT_DICT[47]
        ],
        [
            WAYPOINT_DICT[67],
            WAYPOINT_DICT[77]
        ],
        [
            WAYPOINT_DICT[64],
            WAYPOINT_DICT[71]
        ],
        [
            WAYPOINT_DICT[67],
            WAYPOINT_DICT[71]
        ],
        [
            WAYPOINT_DICT[64],
            WAYPOINT_DICT[77]
        ]
    ]
    return paths

def paths_right_Town01_test():
    
    paths = [
        [
            WAYPOINT_DICT[79],
            WAYPOINT_DICT[103]
        ],
        [
            WAYPOINT_DICT[81],
            WAYPOINT_DICT[105]
        ],
        [
            WAYPOINT_DICT[79],
            WAYPOINT_DICT[105]
        ],
        [
            WAYPOINT_DICT[81],
            WAYPOINT_DICT[103]
        ],
        [
            WAYPOINT_DICT[99],
            WAYPOINT_DICT[84]
        ],
        [
            WAYPOINT_DICT[97],
            WAYPOINT_DICT[86]
        ],
        [
            WAYPOINT_DICT[99],
            WAYPOINT_DICT[86]
        ],
        [
            WAYPOINT_DICT[97],
            WAYPOINT_DICT[84]
        ]
    ]
    return paths

'''
Deprecated
def paths_left_and_right_train():
    
    # # Old longer paths
    # paths = [
    #     # left
    #     [
    #         WAYPOINT_DICT[48],
    #         WAYPOINT_DICT[37]
    #     ],
    #     # right
    #     [
    #         WAYPOINT_DICT[38],
    #         WAYPOINT_DICT[49]
    #     ]
    # ]
    paths = [
        # left
        [
            WAYPOINT_DICT[44],
            WAYPOINT_DICT[39]
        ],
        # right
        [
            WAYPOINT_DICT[42],
            WAYPOINT_DICT[47]
        ]
    ]
    return paths

def paths_left_and_right_test():
    
    # Old longer paths
    # paths = [
    #     # left
    #     [
    #         WAYPOINT_DICT[87],
    #         WAYPOINT_DICT[100]
    #     ],
    #     # right
    #     [
    #         WAYPOINT_DICT[99],
    #         WAYPOINT_DICT[86]
    #     ]
    # ]
    paths = [
        # left
        [
            WAYPOINT_DICT[85],
            WAYPOINT_DICT[98]
        ],
        # right
        [
            WAYPOINT_DICT[94],
            Transform(Location(x=1.5099804401397705, y=278.81, z=1.32), Rotation(yaw=-90.00029754638672))
            # Destination is mid of 84 and 86
            # 84: Transform(Location(x=1.5099804401397705, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
            # 86: Transform(Location(x=1.5099804401397705, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
            
        ]
    ]
    return paths
'''

def get_fixed_short_straight_path_Town01(unseen=False):
    " Returns a list of [start_transform, target_transform]"
    if not unseen:
        return random.choice(paths_straight_Town01_train())
    else:
        return random.choice(paths_straight_Town01_test())

'''Deprecated
def get_fixed_long_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return paths_straight_Town01_train()[1]

def get_random_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return random.choice(paths_straight_Town01_train())
'''

def get_left_right_randomly(unseen = False):
    return random.choice([get_right_turn(unseen=unseen), get_left_turn(unseen=unseen)])

def get_right_turn(unseen = False):
    if unseen:
        return random.choice(paths_right_Town01_test())
    else:
        return random.choice(paths_right_Town01_train())
    
def get_left_turn(unseen = False):
    if unseen:
        return random.choice(paths_left_Town01_test())
    else:
        return random.choice(paths_left_Town01_train())

'''
# Deprecated Helper functions
def get_train_right_turn():
    return paths_left_and_right_train()[1]

def get_test_right_turn():
    return paths_left_and_right_test()[1]

def get_train_left_turn():
    return paths_left_and_right_train()[0]

def get_test_left_turn():
    return paths_left_and_right_test()[0]

def get_train_left_right_randomly():
    return random.choice(paths_left_and_right_train())

def get_test_left_right_randomly():
    return random.choice(paths_left_and_right_test())
'''
