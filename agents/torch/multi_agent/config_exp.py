EXP_CONFIGS = {

    "busy":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_1', 
                'checkpoint': 'ckptDPPO1x9x8_360deg_str_st_fs_1_481584_May130505PM21.pth',
                'ckpt_mode': 'resume',
                'save_freq': 30000,
            }
        },
    
    "360deg_str_st_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_1', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'], 
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_str_st_fs_1_242142_May150844AM31.pth'
            }
        },
    "360deg_str_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_1', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:3'], 
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_1_761842_May170849PM24.pth'
            }
        },
    "360deg_str_st_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'], 
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_str_st_fs_4_30145_May161009AM07.pth'
            }
        },
    "360deg_str_stovrtk_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:0'], 
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
            }
        },
    "360deg_str_st_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_8', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'], 
            }
        },
    "360deg_str_stovrtk_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_8', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'],
            }
        },
    "360deg_str_st_fs_15":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 15,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_15', 
                'save_freq': 30000,
            }
        },
    "360deg_str_stovrtk_fs_15":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 15,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_15', 
                'save_freq': 30000,
            }
        },

    
    "360deg_5dof_st_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_1', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:2'],
            }
        },
    "360deg_5dof_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_1', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:3'],
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_1_243520_May151132AM24.pth'
            }
        },
    "360deg_5dof_st_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'],
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_5dof_st_fs_4_120701_May160911AM57.pth'
            }
        },
    "360deg_5dof_stovrtk_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:3'],
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "360deg_5dof_strandovrtk_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_random_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:3'],
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "360deg_5dof_st_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_8', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'],
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_360deg_5dof_st_fs_8_30011_May150929AM10.pth'
            }
        },
    "360deg_5dof_stovrtk_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_8', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'],
            }
        },
    "360deg_5dof_st_fs_15":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 15,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_15', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_stovrtk_fs_15":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 15,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_15', 
                'save_freq': 30000,
            }
        },

    "360deg_5dof_stovrtk_sa_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 4,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_sa_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:0'],
            }
        },

    "360deg_5dof_st_fs_1_4x8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_agents': 8,
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_1_4x8', 
                'save_freq': 30000,
                'num_workers': 4,
                'device_list': ['cuda:3'],
            }
        },
    "360deg_5dof_st_fs_4_4x8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_agents': 8,
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_4_4x8', 
                'save_freq': 30000,
                'num_workers': 4,
                'device_list': ['cuda:3'],
            }
        },

    "360deg_5dof_st_fs_1_8x8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_agents': 8,
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_1_8x8', 
                'save_freq': 30000,
                'num_workers': 8,
                'device_list': ['cuda:0','cuda:1'],
            }
        },
    "360deg_5dof_st_fs_4_8x8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'num_npc' : 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_agents': 8,
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_4_8x8', 
                'save_freq': 30000,
                'num_workers': 8,
                'device_list': ['cuda:0','cuda:1'],
            }
        },



    "7dim_5dof_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_obs_info_speed_steer_ldist_light',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '7dim_5dof_stovrtk_fs_1', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:2'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'.pth'
            }
        },
    "7dim_5dof_stovrtk_fs_4":{
            "env":{
                'input_type': 'wp_obs_info_speed_steer_ldist_light',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '7dim_5dof_stovrtk_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:2'],
                'ckpt_mode':'resume',
                'checkpoint':'ckptDPPO1x1x1_7dim_5dof_stovrtk_fs_4_30012_May181002AM46.pth'
            }
        },

    "7dim_5dof_strandovrtk_fs_4":{
            "env":{
                'input_type': 'wp_obs_info_speed_steer_ldist_light',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_random_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '7dim_5dof_stovrtk_fs_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:2'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_1_243520_May151132AM24.pth'
            }
        },

}

DEFAULT_CONFIGS = {
    "agent":{
        'num_workers': 1,
        'device_list': ['cuda:0'],  
    },
    "env":{
        'sensor_x_res':'256',
        'sensor_y_res':'256',
        'sensor_tick':'0.0',
        'num_agents': 1, #8,
        'verbose': False,
    }

    # "agent":{
    #     'num_workers': 4,
    #     'device_list': ['cuda:3'],  
    # },
    # "env":{
    #     'sensor_x_res':'164',
    #     'sensor_y_res':'164',
    #     'sensor_tick':'0.5',
    #     'num_agents': 8,
    #     'verbose': False,
    # }

    # "agent":{
    #     'num_workers': 9,
    #     'device_list': ['cuda:1','cuda:2','cuda:3'],  
    # },
    # "env":{
    #     'num_agents': 8, 
    #     'verbose': False,
    # }
}


def override_configs(exp_name,env_cfg,agent_cfg):

    global EXP_CONFIGS, DEFAULT_CONFIGS

    if exp_name not in EXP_CONFIGS:
        raise "Invalid Experiment Configuration!!!"

    exp_cfg = EXP_CONFIGS[exp_name]

    
    for k in DEFAULT_CONFIGS["env"]:
        env_cfg[k] = DEFAULT_CONFIGS["env"][k]

    for k in DEFAULT_CONFIGS["agent"]:
        agent_cfg[k] = DEFAULT_CONFIGS["agent"][k]

    for k in exp_cfg["env"]:
        env_cfg[k] = exp_cfg["env"][k]

    for k in exp_cfg["agent"]:
        agent_cfg[k] = exp_cfg["agent"][k]

    return env_cfg,agent_cfg


"""
tensorboard --logdir=360deg_5dof_st_fs_4_8x8:DPPO1x8x8_360deg_5dof_st_fs_4_8x8,360deg_str_st_fs_1:DPPO1x1x1_360deg_str_st_fs_1,360deg_str_st_fs_4:DPPO1x1x1_360deg_str_st_fs_4,360deg_str_st_fs_8:DPPO1x1x1_360deg_str_st_fs_8,360deg_5dof_st_fs_1:DPPO1x1x1_360deg_5dof_st_fs_1,360deg_5dof_st_fs_4:DPPO1x1x1_360deg_5dof_st_fs_4,360deg_5dof_st_fs_8:DPPO1x1x1_360deg_5dof_st_fs_8,360deg_5dof_st_fs_1_4x8:DPPO1x4x8_360deg_5dof_st_fs_1_4x8,360deg_5dof_st_fs_4_4x8:DPPO1x4x8_360deg_5dof_st_fs_4_4x8 --port=6008

tensorboard --logdir=7dim_5dof_stovrtk_fs_1:DPPO1x1x1_7dim_5dof_stovrtk_fs_1,7dim_5dof_stovrtk_fs_4:DPPO1x1x1_7dim_5dof_stovrtk_fs_4,360deg_str_stovrtk_fs_1:DPPO1x1x1_360deg_str_stovrtk_fs_1,360deg_str_stovrtk_fs_4:DPPO1x1x1_360deg_str_stovrtk_fs_4,360deg_str_stovrtk_fs_8:DPPO1x1x1_360deg_str_stovrtk_fs_8,360deg_5dof_stovrtk_fs_1:DPPO1x1x1_360deg_5dof_stovrtk_fs_1,360deg_5dof_stovrtk_fs_4:DPPO1x1x1_360deg_5dof_stovrtk_fs_4,360deg_5dof_stovrtk_fs_8:DPPO1x1x1_360deg_5dof_stovrtk_fs_8,360deg_5dof_stovrtk_sa_4:DPPO1x1x1_360deg_5dof_stovrtk_sa_4 --port=6007

"""

"""
Log - 

360deg_str_st_fs_1 - 760K - DONE 
360deg_str_st_fs_4 - 38K - Continue @ 30K
360deg_str_st_fs_8 - TODO

360deg_str_stovrtk_fs_1 - 760K - DONE 
360deg_str_stovrtk_fs_4 - 270K - DONE 
360deg_str_stovrtk_fs_8 - TODO

360deg_5dof_st_fs_1 - 240K - DONE
360deg_5dof_st_fs_4 - 137K - Continue @ 120K 
360deg_5dof_st_fs_8 - 72K - Continue @ 60K -

360deg_5dof_stovrtk_fs_1 - 730K - DONE
360deg_5dof_stovrtk_fs_4 - 200K - Continue @ 180K
360deg_5dof_stovrtk_fs_8 - TODO

360deg_5dof_strandovrtk_fs_4 - Restart - v2

360deg_5dof_stovrtk_sa_4 - 500K - DONE 

360deg_5dof_st_fs_1_4x8 - TODO
360deg_5dof_st_fs_4_8x8 - ? @ 690K - v1

7dim_5dof_stovrtk_fs_1 - ? @ 515K - v3
7dim_5dof_stovrtk_fs_4 - Continue @ 30K - v4
7dim_5dof_strandovrtk_fs_4 - Restart - v5
"""