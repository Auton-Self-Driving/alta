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
    
    "test":{
            "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':3000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'dist_for_success':4.0
                },
            "agent":{
                'save_suffix': 'test', 
                'save_freq': 30000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },

    ### Full Action Space Variable Frame skip

    "360deg_str_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1500000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_1', 
                'save_freq': 30000,
                'num_workers': 7,
                'device_list': ['cuda:3'], 
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_1_761842_May170849PM24.pth'
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
                    'max_num_steps':3000000,
                    'num_agents':1,
                    'npc_reset_freq':1
                },
                "agent":{
                    'save_suffix': '360deg_str_stovrtk_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_str_stovrtk_fs_16":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 16,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':300000,
                "num_agents":1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_str_stovrtk_fs_16', 
                'save_freq': 30000,
                'num_workers': 14,
                'device_list': ['cuda:0','cuda:1'],
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
                'max_num_steps':10000000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_1', 
                'save_freq': 30000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_1_243520_May151132AM24.pth'
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
                'max_num_steps':3000000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_4', 
                'save_freq': 30000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "360deg_5dof_stovrtk_fs_4_seed_2":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':3000000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_4_seed_2', 
                'save_freq': 30000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "360deg_5dof_stovrtk_fs_4_seed_3":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':3000000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_4_seed_3', 
                'save_freq': 30000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "360deg_5dof_stovrtk_fs_16":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 16,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':300000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_16', 
                'save_freq': 30000,
                'num_workers': 14,
                'device_list': ['cuda:2','cuda:3'],
            }
        },

    "360deg_5dof_stovrtk_fs_1_str_pen":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':9900000,
                'num_agents':1,
                'npc_reset_freq':1,
                'steer_penalty_coeff':10
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtk_fs_1_str_pen', 
                'save_freq': 30000,
                'num_workers': 14,
                'device_list': ['cuda:1','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x12x1_360deg_5dof_stovrtk_fs_1_seed_3_570938_Aug160521PM38.pth'
            }
        },

    ### Contant Speed Autopilot

    "360deg_5dof_20_spd_stovrtk_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':700000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_const_speed':20.0,
                },
                "agent":{
                    'save_suffix': '360deg_5dof_20_spd_stovrtk_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_5dof_20_spd_stovrtk_fs_4_seed_2":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':3000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_const_speed':20.0,
                    'dist_for_success':4.0
                },
                "agent":{
                    'save_suffix': '360deg_5dof_20_spd_stovrtk_fs_4_seed_2', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
        
    "360deg_5dof_20_spd_stovrtk_fs_16":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 16,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':400000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_const_speed':20.0,

                },
                "agent":{
                    'save_suffix': '360deg_5dof_20_spd_stovrtk_fs_16', 
                    'save_freq': 30000,
                    'num_workers': 14, 
                    'device_list': ['cuda:0','cuda:1'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },

    "360deg_5dof_20_spd_stovrtk_fs_4_0K":{ # Verification. Dayum not zero. Woah
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':3000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_steps': 0,
                    'autopilot_const_speed':20.0,
                    'dist_for_success':4.0
                },
                "agent":{
                    'save_suffix': '360deg_5dof_20_spd_stovrtk_fs_4_0K', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_5dof_20_spd_stovrtk_fs_4_10K":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':3000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_steps': 10000,
                    'autopilot_const_speed':20.0,
                    'dist_for_success':4.0
                },
                "agent":{
                    'save_suffix': '360deg_5dof_20_spd_stovrtk_fs_4_10K', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },

    "360deg_str_20_spd_stovrtk_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'merged_speed_scaled_tanh', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':700000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_const_speed':20.0,

                },
                "agent":{
                    'save_suffix': '360deg_str_20_spd_stovrtk_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 14, 
                    'device_list': ['cuda:0','cuda:1'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_str_20_spd_stovrtk_fs_16":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'merged_speed_scaled_tanh', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 16,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':300000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_const_speed':20.0,
                },
                "agent":{
                    'save_suffix': '360deg_str_20_spd_stovrtk_fs_16', 
                    'save_freq': 30000,
                    'num_workers': 14, 
                    'device_list': ['cuda:2','cuda:3'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },

    ### MDP Connectivity Hypothesis
    "360deg_5dof_20_spd_stovrtkcls_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake_closeby',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':3000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'const_speed',
                    'autopilot_const_speed':20.0,
                },
                "agent":{
                    'save_suffix': '360deg_5dof_20_spd_stovrtkcls_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_5dof_stovrtkcls_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake_closeby',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':700000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_stovrtkcls_fs_4', 
                'save_freq': 30000,
                'num_workers': 7,
                'device_list': ['cuda:2'], 
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "360deg_5dof_ppo_part_steer_final_stovrtkcls_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake_closeby',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':700000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'PPO_part_steer_final',
                    # 'autopilot_ckpt' : '360deg_5dof_steer_only_stovrtk_fs_4/ckptDPPO1x14x1_360deg_5dof_steer_only_stovrtk_fs_4_1260731_Jun300846AM45.pth',
                    'autopilot_ckpt' : '360deg_5dof_stovrtk_fs_1/ckptDPPO1x12x1_360deg_5dof_stovrtk_fs_1_1472372_Aug031203AM37.pth',
                },
                "agent":{
                    'save_suffix': '360deg_5dof_ppo_part_steer_final_stovrtkcls_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 7, 
                    'device_list': ['cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },

    ### Steer autopilots
    "360deg_5dof_ppo_steer_stovrtk_fs_4_90_autopilot":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':1000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'PPO_steer',
                    # 'autopilot_ckpt' : '360deg_5dof_steer_only_stovrtk_fs_4/ckptDPPO1x14x1_360deg_5dof_steer_only_stovrtk_fs_4_1260731_Jun300846AM45.pth',
                    'autopilot_ckpt' : '360deg_5dof_20_spd_stovrtk_fs_4_seed_2/ckptDPPO1x20x1_360deg_5dof_20_spd_stovrtk_fs_4_seed_2_2221342_Aug270503PM40.pth',
                },
                "agent":{
                    'save_suffix': '360deg_5dof_ppo_steer_stovrtk_fs_4_90_autopilot', 
                    'save_freq': 30000,
                    'num_workers': 20,
                    'device_list': ['cuda:1','cuda:2','cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_5dof_ppo_steer_stovrtk_fs_4_60_autopilot":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':1000000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'PPO_steer',
                    # 'autopilot_ckpt' : '360deg_5dof_steer_only_stovrtk_fs_4/ckptDPPO1x14x1_360deg_5dof_steer_only_stovrtk_fs_4_1260731_Jun300846AM45.pth',
                    'autopilot_ckpt' : '360deg_5dof_20_spd_stovrtk_fs_4_seed_2/ckptDPPO1x20x1_360deg_5dof_20_spd_stovrtk_fs_4_seed_2_180112_Aug251023PM42.pth',
                },
                "agent":{
                    'save_suffix': '360deg_5dof_ppo_steer_stovrtk_fs_4_60_autopilot', 
                    'save_freq': 30000,
                    'num_workers': 7,
                    'device_list': ['cuda:3'], #['cuda:1','cuda:2','cuda:3'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_5dof_ppo_steer_stovrtk_fs_4_60_autopilot_100K":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'straight_overtake',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':2000000,
                'num_agents':1,
                'npc_reset_freq':1,
                'autopilot_steps': 100000,
                'autopilot_type' : 'PPO_steer',
                # 'autopilot_ckpt' : '360deg_5dof_steer_only_stovrtk_fs_4/ckptDPPO1x14x1_360deg_5dof_steer_only_stovrtk_fs_4_1260731_Jun300846AM45.pth',
                'autopilot_ckpt' : '360deg_5dof_20_spd_stovrtk_fs_4_seed_2/ckptDPPO1x20x1_360deg_5dof_20_spd_stovrtk_fs_4_seed_2_180112_Aug251023PM42.pth',
            },
            "agent":{
                'save_suffix': '360deg_5dof_ppo_steer_stovrtk_fs_4_60_autopilot_100K', 
                'save_freq': 30000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
            }
        },
    
    "360deg_5dof_ppo_part_steer_final_stovrtk_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':700000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'PPO_part_steer_final',
                    # 'autopilot_ckpt' : '360deg_5dof_steer_only_stovrtk_fs_4/ckptDPPO1x14x1_360deg_5dof_steer_only_stovrtk_fs_4_1260731_Jun300846AM45.pth',
                    'autopilot_ckpt' : '360deg_5dof_stovrtk_fs_1/ckptDPPO1x12x1_360deg_5dof_stovrtk_fs_1_1472372_Aug031203AM37.pth',
                },
                "agent":{
                    'save_suffix': '360deg_5dof_ppo_part_steer_final_stovrtk_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 10, 
                    'device_list': ['cuda:0','cuda:1'],
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    "360deg_5dof_ppo_part_steer_interm_stovrtk_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':700000,
                    'num_agents':1,
                    'npc_reset_freq':1,
                    'autopilot_type' : 'PPO_part_steer_intermediate',
                    # 'autopilot_ckpt' : '360deg_5dof_steer_only_stovrtk_fs_4/ckptDPPO1x14x1_360deg_5dof_steer_only_stovrtk_fs_4_1260731_Jun300846AM45.pth',
                    'autopilot_ckpt' : '360deg_5dof_stovrtk_fs_1/ckptDPPO1x12x1_360deg_5dof_stovrtk_fs_1_1472372_Aug031203AM37.pth',
                },
                "agent":{
                    'save_suffix': '360deg_5dof_ppo_part_steer_interm_stovrtk_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 10, 
                    'device_list': ['cuda:3','cuda:2'],#'cuda:2','cuda:3'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },
    
    # TODO: Perform this for FS = 16

    # -----------------------------------------------------

    "360deg_5dof_no_crash_reg_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'no_crash_regular',
                'num_npc': 1,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':10000000,
                'sample_npc':True,
                'num_npc_lower_threshold':20,
                'num_npc_upper_threshold':20,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_5dof_no_crash_reg_fs_4', 
                'save_freq': 50000,
                'num_workers': 13,
                'device_list': ['cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "no_crash_reg_360deg_5dof_40_spd_fs_1_bad_auto":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'no_crash_regular',
                'num_npc': 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':10000000,
                'sample_npc':True,
                'num_npc_lower_threshold':20,
                'num_npc_upper_threshold':30,
                'terminate_on_light':False,
                'npc_reset_freq':1,
                'autopilot_type' : 'const_speed',
                'autopilot_const_speed':40.0,
                'slow_no_light_autopilot':True,
                'custom_offroad_check':False
            },
            "agent":{
                'save_suffix': 'no_crash_reg_360deg_5dof_40_spd_fs_1_bad_auto', 
                'save_freq': 50000,
                'num_workers': 20,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },




    ### TODO: Increase traj_frame_horizon and test with steer autopilots

    ### Disrete Throttle
    "360deg_5dof_disc_thrt_stovrtk_fs_4":{
                "env":{
                    'input_type': 'wp_360_obstacle_speed_steer',
                    'action_type': 'cubic_bezier_5dof_disc_thrt', 
                    'scenarios' : 'straight_overtake',
                    'num_npc': 1,
                    'frame_skip': 4,
                    'traj_frame_horizon':30, 
                    'sticky_temporal_action_frames': 1,
                    'max_num_steps':5000000,
                    'num_agents':1,
                    'npc_reset_freq':1
                },
                "agent":{
                    'save_suffix': '360deg_5dof_disc_thrt_stovrtk_fs_4', 
                    'save_freq': 30000,
                    'num_workers': 14, 
                    'device_list': ['cuda:1','cuda:2','cuda:3'],#'cuda:2','cuda:3'], 
                    # 'ckpt_mode':'resume',
                    # 'checkpoint':'ckptDPPO1x1x1_360deg_str_stovrtk_fs_4_60334_May151137PM15.pth'
                }
            },

    ### Single waypoint action space
    "360deg_wp_stovrtk_fs_1":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'speed_wp', 
                'scenarios' : 'straight_overtake',
                'num_npc' : 1,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_agents':1,
                'npc_reset_freq':1
            },
            "agent":{
                'save_suffix': '360deg_wp_stovrtk_fs_1', 
                'save_freq': 30000,
                'num_workers': 6,
                'device_list': ['cuda:2'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_wp_stovrtk_fs_4_60244_May300916AM52.pth'
            }
        },

    ##### Self-play Ablations ####
    #### Num Agents
    "360deg_str_st_fs_1_ag_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_npc' : 0,
                'num_agents':4,
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_1_ag_4', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'], 
            }
        },
    "360deg_str_st_fs_1_ag_8":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_npc' : 0,
                'num_agents':8,
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_1_ag_8', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'], 
            }
        },
    "360deg_str_st_fs_1_ag_16":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'merged_speed_scaled_tanh', 
                'scenarios' : 'straight',
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':1000000,
                'num_npc' : 0,
                'num_agents':16,
            },
            "agent":{
                'save_suffix': '360deg_str_st_fs_1_ag_16', 
                'save_freq': 30000,
                'num_workers': 1,
                'device_list': ['cuda:1'], 
            }
        },


    # No crash Experiments

    # No Crash Empty - FS 4 - Spd 20
    "no_crash_empty_360deg_5dof_20_spd_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'no_crash_empty',
                'num_npc': 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':10000000,
                'num_npc':0,
                'terminate_on_light':False,
                'npc_reset_freq':10000000,
                'autopilot_type' : 'const_speed',
                'autopilot_const_speed':20.0,
                'custom_offroad_check':False
            },
            "agent":{
                'save_suffix': 'no_crash_empty_360deg_5dof_20_spd_fs_4', 
                'save_freq': 50000,
                'num_workers': 24,
                'device_list': ['cuda:0','cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "no_crash_empty_360deg_nospd_5dof_10_spd_fs_4":{
            "env":{
                'input_type': 'wp_360_obstacle_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'no_crash_empty',
                'num_npc': 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':10000000,
                'num_npc':0,
                'terminate_on_light':False,
                'npc_reset_freq':10000000,
                'autopilot_type' : 'const_speed',
                'autopilot_const_speed':10.0,
                'custom_offroad_check':False
            },
            "agent":{
                'save_suffix': 'no_crash_empty_360deg_nospd_5dof_10_spd_fs_4', 
                'save_freq': 50000,
                'num_workers': 24,
                'device_list': ['cuda:0','cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },
    "no_crash_empty_360deg_nospdstr_5dof_10_spd_fs_4":{ # Resume from 870K
            "env":{
                'input_type': 'wp_360_obstacle',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'no_crash_empty',
                'num_npc': 0,
                'frame_skip': 4,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':10000000,
                'num_npc':0,
                'terminate_on_light':False,
                'npc_reset_freq':10000000,
                'autopilot_type' : 'const_speed',
                'autopilot_const_speed':10.0,
                'custom_offroad_check':False
            },
            "agent":{
                'save_suffix': 'no_crash_empty_360deg_nospdstr_5dof_10_spd_fs_4', 
                'save_freq': 50000,
                'num_workers': 24,
                'device_list': ['cuda:0','cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
            }
        },

        "test":{
            "env":{
                'input_type': 'wp_360_obstacle_speed_steer',
                'action_type': 'cubic_bezier_5dof', 
                'scenarios' : 'no_crash_empty',
                'num_npc': 0,
                'frame_skip': 1,
                'traj_frame_horizon':30, 
                'sticky_temporal_action_frames': 1,
                'max_num_steps':10000000,
                'num_npc':0,
                'terminate_on_light':False,
                'npc_reset_freq':1,
                'autopilot_type' : 'const_speed',
                'autopilot_const_speed':20.0,
                'custom_offroad_check':False
            },
            "agent":{
                'save_suffix': 'test', 
                'save_freq': 50000,
                'num_workers': 15,
                'device_list': ['cuda:1','cuda:2','cuda:3'],
                # 'ckpt_mode':'resume',
                # 'checkpoint':'ckptDPPO1x1x1_360deg_5dof_stovrtk_fs_4_60172_May151043AM25.pth'
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
        'num_agents': 1,
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

# Self-play
tensorboard --logdir=ag1:<> ,ag4:DPPO1x1x4_360deg_str_st_fs_1_ag_4,ag8:DPPO1x1x8_360deg_str_st_fs_1_ag_8 --port=6006

tensorboard --logdir=bez4:DPPO1x6x1_360deg_5dof_stovrtk_fs_4,bez1:DPPO1x6x1_360deg_5dof_stovrtk_fs_1,steer4:DPPO1x6x1_360deg_str_stovrtk_fs_4,steer1:DPPO1x6x1_360deg_str_stovrtk_fs_1 --port=6006

tensorboard --logdir=bez4:DPPO1x6x1_360deg_5dof_st_fs_4,steer4:DPPO1x6x1_360deg_str_st_fs_4,steer1:DPPO1x6x1_360deg_str_st_fs_1 --port=6006

Log - 

360deg_str_st_fs_1 - 1M DONE
360deg_str_st_fs_4 - 1M DONE
360deg_str_st_fs_8 -

360deg_str_stovrtk_fs_1 - v3
360deg_str_stovrtk_fs_4 - v4 
360deg_str_stovrtk_fs_8 - 

360deg_5dof_st_fs_1 - 
360deg_5dof_st_fs_4 - 690K DONE
360deg_5dof_st_fs_8 - 

360deg_5dof_stovrtk_fs_1 - v5
360deg_5dof_stovrtk_fs_4 - v2
360deg_5dof_stovrtk_fs_8 - 

360deg_wp_st_fs_1 - 
360deg_wp_st_fs_4 - 390K DONE
360deg_wp_st_fs_8 - 

360deg_wp_stovrtk_fs_1 -
360deg_wp_stovrtk_fs_4 - s2
360deg_wp_stovrtk_fs_8 - 

360deg_5dof_stovrtkrnd_fs_4 -  

360deg_str_st_fs_1_ag_4 - 1M 
360deg_str_st_fs_1_ag_8 - 390K
360deg_str_st_fs_1_ag_16 - 

"""