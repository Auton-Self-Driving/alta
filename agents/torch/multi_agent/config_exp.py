EXP_CONFIGS = {

    "360deg_str_st_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_1', 
                'save_freq': 30000,
            }
        },
    "360deg_str_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_1', 
                'save_freq': 30000,
            }
        },
    "360deg_str_st_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_4', 
                'save_freq': 30000,
            }
        },
    "360deg_str_stovrtk_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_4', 
                'save_freq': 30000,
            }
        },
    "360deg_str_st_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_8', 
                'save_freq': 30000,
            }
        },
    "360deg_str_stovrtk_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_8', 
                'save_freq': 30000,
            }
        },
    "360deg_str_st_fs_15":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
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
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_1', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_1', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_st_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_4', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_stovrtk_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_4', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_st_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_st_fs_8', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_stovrtk_fs_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'frame_skip': 8,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_8', 
                'save_freq': 30000,
            }
        },
    "360deg_5dof_st_fs_15":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight',
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
}

GLOBAL_CONFIGS = {
    "agent":{
        'num_workers': 1,
        'device_list': ['cuda:1'],  
    },
    "env":{
        'num_agents': 1, #8,
        'verbose': False,
    }
}


def override_configs(exp_name,env_cfg,agent_cfg):

    global EXP_CONFIGS, GLOBAL_CONFIGS

    if exp_name not in EXP_CONFIGS:
        raise "Invalid Experiment Configuration!!!"

    exp_cfg = EXP_CONFIGS[exp_name]

    for k in exp_cfg["env"]:
        env_cfg[k] = exp_cfg["env"][k]

    for k in GLOBAL_CONFIGS["env"]:
        env_cfg[k] = GLOBAL_CONFIGS["env"][k]

    for k in exp_cfg["agent"]:
        agent_cfg[k] = exp_cfg["agent"][k]

    for k in GLOBAL_CONFIGS["agent"]:
        agent_cfg[k] = GLOBAL_CONFIGS["agent"][k]

    return env_cfg,agent_cfg
