import argparse
import sys, os

sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
from environment.carla_9_4.config import ConfigManager
from train_measurements_sac_run import run_sac
from train_measurements_ppo_run import run_ppo
from train_vae_ppo_run import run_ppo_vae
from test_pid import test_pid_method
from train_measurements_dqn_run import run_dqn
# from c51new import run_c51
from iqn import run_iqn
from generate_iqn_plots import run_iqn_plots
from iqn_vae import run_iqn_vae


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser to run all deep RL algorithms')
    parser.add_argument('--algo',dest='algo',type=str,required=True, help='Algo: PPO or SAC or DQN or PID_TUNE')
    parser.add_argument('--test', dest='test', action='store_true', help='Enable testing.')
    parser.add_argument('--validation', dest='validation', action='store_true', help='Enable validation.')
    parser.add_argument('--test-trails', dest='test_trails', type=int, default=5, help='No of different test trials.')
    parser.add_argument('--city_name',dest='city_name',type=str, default='Town01', help='Carla Town.')
    parser.add_argument('--vae_model_path',dest='vae_model_path',type=str, default='/zfsauton2/home/hiteshar/research/alta/agents/tf/trained_models/ae_model.json', help='VAE Model path.')
    parser.add_argument('--agent_model_path',dest='agent_model_path',type=str, default=None, help='Agent Model path.')
    parser.add_argument('--input-type', dest='input_type', type=str, default='wp', help='Observation type: "wp", "wp_constant", "wp_noise" or "wp_vae"')
    parser.add_argument('--scenarios', dest='scenarios', type=str, default='navigation', help='CARLA Scenarios type: "straight", "curved", "navigation" or "dynamic_navigation"')
    parser.add_argument('--lr',dest='lr',type=float,default=3e-4)
    parser.add_argument('--ent-coef',dest='ent_coef',type=float,default=0.005, help='Entropy term for PPO runs.')
    parser.add_argument('--buffer-size',dest='buffer_size',type=int,default=50000)
    parser.add_argument('--run-id',dest='run_id',type=str, required=True, help='Unique identifier for the run. It is appended to log directory name.')
    
    parser.add_argument('--network',dest='network',type=str,default='2_layer', help='network: 1_layer, 2_layer, CustomPolicy1 or CustomPolicy2.')
    parser.add_argument('--alpha',dest='alpha',type=float,default=1., help='quantile for iqn runs')
    parser.add_argument('--num-atoms',dest='num_atoms',type=int,default=8, help='num atoms for dlr')
    
    parser.add_argument('--steer-penalty-coeff',dest='steer_penalty_coeff',type=float,default=0, help='Coefficient of steer penalty in reward.')
    parser.add_argument('--noise-dim',dest='noise_dim',type=int,default=1, help='Dimension of noise vector.')
    parser.add_argument('--carla-gpu',dest='carla_gpu',type=str,default='0')
    parser.add_argument('--code-gpu',dest='code_gpu',type=str,default='0')
    parser.add_argument('--base-log-dir',dest='base_log_dir',type=str, required=True, help='base log directory, Eg: /zfsauton2/home/hiteshar/research/alta-logs/new_env/sac_runs1/')
    parser.add_argument('--timesteps',dest='timesteps',type=int,default=1000000, help='total timesteps to train')
    parser.add_argument('--finetune-vae', dest='finetune_vae', action='store_true', help='Whether to finetune vae')
    parser.add_argument('--train-vae', dest='train_vae', action='store_true', help='Whether to train vae from scratch.')
    parser.add_argument('--num-npc',dest='num_npc',type=int,default=0, help='number of other vehicles')
    parser.add_argument('--videos', dest='videos', action='store_true', help='Whether to save videos')
    parser.add_argument('--const-collision-penalty',dest='const_collision_penalty',type=float,default=0.0, help='Constant penalty for collision.')
    parser.add_argument('--collision-penalty-speed-coeff',dest='collision_penalty_speed_coeff',type=float,default=0.0, help='Speed coefficient for speed-proportional collision penalty.')
    parser.add_argument('--const-light-penalty',dest='const_light_penalty',type=float,default=0.0, help='Constant penalty for running traffic light.')
    parser.add_argument('--light-penalty-speed-coeff',dest='light_penalty_speed_coeff',type=float,default=0.0, help='Speed-proportional penalty for running light.')
    # parser.add_argument('--enable-brake', dest='enable_brake', action='store_true', help='Whether to enable brake action')
    parser.add_argument('--fs',dest='frame_skip',type=int,default=1, help='Number of frame skip (default:1)')
    parser.add_argument('--n-steps',dest='n_steps',type=int,default=500, help='Number of steps in trajectory for PPO.')
    parser.add_argument('--no-epochs',dest='no_epochs',type=int,default=4, help='Number of epochs to optimize the minibatch for PPO.')
    parser.add_argument('--no-minibatches',dest='no_minibatches',type=int,default=4, help='Number of minibatches for PPO.')
    parser.add_argument('--clip',dest='clip',type=float,default=0.2, help='Clip parameter for PPO.')
    parser.add_argument('--disable-semantic', dest='disable_semantic', action='store_true', help='Whether to disable semantic segmentation camera and enable RGB camera. (semantic is enabled by default).')
    parser.add_argument('--disable-collision', dest='disable_collision', action='store_true', help='Whether to disable collision for episode done condition.')
    parser.add_argument('--disable-traffic-light', dest='disable_traffic_light', action='store_true', help='Whether to disable traffic light.')
    parser.add_argument('--disable-obstacle-info', dest='disable_obstacle_info', action='store_true', help='Whether to disable obstacle detector.')
    parser.add_argument('--enable-static', dest='enable_static', action='store_true', help='Whether to enable max static steps for episode done condition.')
    parser.add_argument('--static-steps',dest='static_steps',type=int,default=1000, help='Max no of static steps.')
    parser.add_argument('--ae-lr',dest='ae_lr',type=float,default=1e-4)
    parser.add_argument('--disable-pid-fs',dest='disable_pid_fs', action='store_true', help='Disable using pid within each frameskip. (Default way is to use pid within frameskip)')
    parser.add_argument('--fstack',dest='frame_stack',type=int,default=1, help='Input frame stack size (default:1)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--dqn-param-noise',dest='param_noise', action='store_true', help='Whether to enable param_noise in dqn.')
    parser.add_argument('--dqn-prioritized-replay',dest='prioritized_replay', action='store_true', help='Whether to enable prioritized replay in dqn.')
    parser.add_argument('--full-tb-log',dest='full_tensorboard_log', action='store_true', help='Whether to enable full tensorboard logging.')
    parser.add_argument('--clip-reward',dest='clip_reward', action='store_true', help='Whether to clip reward.')
    parser.add_argument('--train-buffer', dest='train_buffer', action='store_true', help='Train using replay buffer.')
    parser.add_argument('--special-sample', dest='special_sample', action='store_true', help='Sample t=0, 1, 2 transitions more.')
    parser.add_argument('--target-freq',dest='target_freq',type=int,default=2000, help='Target network update frequency.')
    parser.add_argument('--reward-norm', dest='reward_norm', type=int, default=1, help='A constant factor to normalize the reward.')
    parser.add_argument('--success-reward', dest='success_reward', type=int, default=0, help='Constant reward to add on success.')
    parser.add_argument('--dqn-n-step',dest='dqn_n_step',type=int,default=1, help='n in n-step DQN. n=1 corresponds to standard DQN.')
    parser.add_argument('--constant-reward', dest='constant_reward', type=int, default=0, help='Constant reward to add on each time step.')

    return parser.parse_args()
def main(args):

    args = parse_arguments()
    return args

def create_sac_prefix(args):

    base = 'algo_' + args.algo \
        + '_input_' + args.input_type \
        + '_network_' + str(args.network) \
        + '_lr_' + str(args.lr)  \
        + '_buffer_' + str(args.buffer_size) \
        + '_' + args.scenarios \
    
    prefix = base + '_runid_' + args.run_id + '/'
    base_prefix = base + '/'
    return base_prefix, prefix

def create_ppo_prefix(args):

    if args.finetune_vae:
        vae = "_finetune_vae"
    elif args.train_vae:
        vae = "_train_vae"
    else:
        vae = ""
    
    if args.num_npc != 0:
        num_npc_str = '_npc_' + str(args.num_npc)
    else:
        num_npc_str = ""

    if args.buffer_size != 0:
        buffer_size_str = '_buffer_' + str(args.buffer_size)
    else:
        buffer_size_str = ""

    if args.reward_norm != 1:
        reward_norm_str = '_rew_norm_' + str(args.reward_norm)
    else:
        reward_norm_str = ""
    
    if args.success_reward != 0:
        success_reward_str = '_successr_' + str(args.success_reward)
    else:
        success_reward_str = ""

    if args.constant_reward != 0:
        constant_reward_str = '_constantr_' + str(args.constant_reward)
    else:
        constant_reward_str = ""
    
    if args.const_collision_penalty != 0:
        const_collision_penalty_str = '_col_' + str(args.const_collision_penalty)
    else:
        const_collision_penalty_str = ""
    
    if args.collision_penalty_speed_coeff != 0:
        collision_penalty_speed_coeff_str = '_col_sp_' + str(args.collision_penalty_speed_coeff)
    else:
        collision_penalty_speed_coeff_str = ""

    if args.const_light_penalty != 0:
        const_light_penalty_str = '_light_' + str(args.const_light_penalty)
    else:
        const_light_penalty_str = ""

    if args.light_penalty_speed_coeff != 0:
        light_penalty_speed_coeff_str = '_light_sp_' + str(args.light_penalty_speed_coeff)
    else:
        light_penalty_speed_coeff_str = ""

    if args.steer_penalty_coeff != 0:
        steer_penalty_coeff_str = '_steer_pen_' + str(args.steer_penalty_coeff)
    else:
        steer_penalty_coeff_str = ""

    # if args.enable_brake != False:
    #     enable_brake_str = '_brake'
    # else:
    #     enable_brake_str = ''

    if args.disable_collision != False:
        disable_collision_str = "_disable_collision_"
    else:
        disable_collision_str = ''

    if args.disable_traffic_light != False:
        disable_traffic_light_str = "_disable_light_"
    else:
        disable_traffic_light_str = ''

    if args.disable_obstacle_info != False:
        disable_obstacle_info_str = "_disable_obs_"
    else:
        disable_obstacle_info_str = ''

    if args.enable_static != False:
        enable_static_str = "_enable_static_" + str(args.static_steps) + "_"
    else:
        enable_static_str = ''

    if args.target_freq != 2000:
        target_freq_str = "_target_freq_" + str(args.target_freq) + "_"
    else:
        target_freq_str = ''

    if args.ent_coef != 0.005:
        ent_coef_str = '_ent_' + str(args.ent_coef)
    else:
        ent_coef_str = ''
    
    if args.input_type == "wp_noise" or args.input_type == "wp_obs_bool_noise":
        input_type = args.input_type + str(args.noise_dim)
    else:
        input_type = args.input_type
    
    if args.frame_skip != 1:
        frame_skip_str = '_fs_' + str(args.frame_skip)
    else:
        frame_skip_str = ''
    
    if args.frame_stack != 1:
        frame_stack_str = '_fstack_' + str(args.frame_stack)
    else:
        frame_stack_str = ''
    
    if args.n_steps != 500:
        n_steps_str = '_n_' + str(args.n_steps)
    else:
        n_steps_str = ''
    
    if args.dqn_n_step != 1:
        dqn_n_step_str = '_dqn_n_' + str(args.dqn_n_step)
    else:
        dqn_n_step_str = ''

    if args.agent_model_path is not None:
        use_pretrained_agent_str = '_pretrained_agent_'
    else:
        use_pretrained_agent_str = ''

    if args.ae_lr != 1e-4:
        ae_lr_str = '_ae_lr_' + str(args.ae_lr)
    else:
        ae_lr_str = ''

    if args.clip_reward:
        clip_reward_str = '_clip_reward_'
    else:
        clip_reward_str = ''
    
    
    if args.param_noise:
        param_noise_str = '_param_noise_'
    else:
        param_noise_str = ''
    
    if args.special_sample:
        special_sample_str = '_special_sample_'
    else:
        special_sample_str = ''
    
    if args.prioritized_replay:
        prioritized_replay_str = '_prioritized_replay_'
    else:
        prioritized_replay_str = ''


    if args.disable_pid_fs:
        disable_pid_fs_str = '_disable_pid_fs_'
    else:
        disable_pid_fs_str = ''

    noptepochs_str = '_epochs_{}_'.format(args.no_epochs)
    clip_str = '_clip_{}_'.format(args.clip)
    no_minibatches_str = '_mb_{}_'.format(args.no_minibatches)

    prefix = 'algo_' + args.algo \
        + '_input_' + input_type \
        + '_network_' + str(args.network) \
        + '_lr_' + str(args.lr)  \
        + ae_lr_str \
        + '_' + args.scenarios \
        + use_pretrained_agent_str \
        + num_npc_str \
        + buffer_size_str \
        + disable_collision_str \
        + disable_traffic_light_str \
        + disable_obstacle_info_str \
        + enable_static_str \
        + target_freq_str \
        + const_collision_penalty_str \
        + collision_penalty_speed_coeff_str \
        + const_light_penalty_str \
        + light_penalty_speed_coeff_str \
        + steer_penalty_coeff_str \
        + ent_coef_str \
        + frame_skip_str \
        + frame_stack_str \
        + disable_pid_fs_str \
        + clip_reward_str \
        + param_noise_str \
        + special_sample_str \
        + prioritized_replay_str \
        + n_steps_str \
        + dqn_n_step_str \
        + vae \
        + reward_norm_str \
        + success_reward_str \
        + constant_reward_str \

    if args.algo == "PPO":
        prefix = prefix + noptepochs_str \
        + clip_str \
        + no_minibatches_str \


    prefix = prefix + '_runid_' + args.run_id + '/'
    return prefix

def get_drl_prefix(args): 
    prefix = 'algo_' + args.algo \
        + '_input_' + str(args.input_type) \
        + '_scenario_' + str(args.scenarios)  \
        + '_npcs_' + str(args.num_npc)  \
        + '_lr_' + str(args.lr)  \
        + '_alpha_' + str(args.alpha) \
        + '_network_' + str(args.network) \
        + '_targetfreq_' + str(args.target_freq) \
        + '_rewardnorm_' + str(args.reward_norm) \
        + '_lightpenalty_' + str(args.const_light_penalty) \
        + '_frameskip_' + str(args.frame_skip) \
        + '_runid_' + str(args.run_id) 
    return prefix

def extract_prefix(args):
    prefix = args.agent_model_path.split('/')[-2]
    return prefix

if __name__ == '__main__':
    args = main(sys.argv)
    print("args", args)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.code_gpu)

    config = ConfigManager(algo=args.algo)
    config.config["carla_gpu"] = str(args.carla_gpu)
    config.config["steer_penalty_coeff"] = args.steer_penalty_coeff
    config.config["input_type"] = args.input_type
    config.config["scenarios"] = args.scenarios
    config.config["train_vae"] = (args.train_vae or args.finetune_vae)
    config.config["noise_dim"] = args.noise_dim
    config.config["num_npc"] = args.num_npc
    config.config["videos"] = args.videos
    config.config["const_collision_penalty"] = args.const_collision_penalty
    config.config["collision_penalty_speed_coeff"] = args.collision_penalty_speed_coeff
    config.config["const_light_penalty"] = args.const_light_penalty
    config.config["light_penalty_speed_coeff"] = args.light_penalty_speed_coeff
    # config.config["enable_brake"] = args.enable_brake
    config.config["frame_skip"] = args.frame_skip
    config.config["frame_stack_size"] = args.frame_stack
    config.config["semantic"] = not args.disable_semantic
    config.config["disable_collision"] = args.disable_collision
    config.config["disable_traffic_light"] = args.disable_traffic_light
    config.config["disable_obstacle_info"] = args.disable_obstacle_info
    config.config["enable_static"] = args.enable_static
    config.config["max_static_steps"] = args.static_steps
    config.config["city_name"] = args.city_name
    config.config["testing"] = args.test
    config.config["use_pid_in_frame_skip"] = not args.disable_pid_fs
    config.config["verbose"] = args.verbose
    config.config["clip_reward"] = args.clip_reward
    config.config["reward_normalize_factor"] = args.reward_norm
    config.config["success_reward"] = args.success_reward
    config.config["constant_positive_reward"] = args.constant_reward

    try:
        if args.algo == "SAC":
            base_prefix, prefix = create_sac_prefix(args)
            print("prefix", prefix)
            run_sac(args, prefix, base_prefix, config)
        elif args.algo == "PPO":
            if not args.test:
                prefix = create_ppo_prefix(args)
            else:
                prefix = extract_prefix(args)
            print("prefix", prefix)
            if args.input_type in ['wp', 'wp_noise', 'wp_obs_dist', 'wp_obs_bool', 'wp_obs_bool_noise', 'wp_ldist_goal',
                                   'wp_obs_bool_speed_steer_goal_light', 'wp_obs_info_speed_steer_ldist_goal_light']:
                run_ppo(args, prefix, config)
            elif args.input_type in ['wp_vae', 'wp_vae_speed_steer_goal', 'wp_vae_speed_steer_ldist_goal_light']:
                run_ppo_vae(args, prefix, config)
            else:
                print("specify correct input_type: wp, wp_vae")
                print("exiting")
        elif args.algo == "PID_TUNE":
            prefix = create_ppo_prefix(args)
            test_pid_method(args, prefix, config)
        elif args.algo == "DQN":
            if args.test or args.train_buffer:
                prefix = extract_prefix(args) 
            else:
                prefix = create_ppo_prefix(args)
            run_dqn(args, prefix, config)
        # elif args.algo == 'C51': 
        #     # prefix = extract_prefix(args) 
        #     prefix = get_drl_prefix(args)
        #     run_c51(args, prefix, config)
        elif args.algo == 'IQN': 
            # prefix = extract_prefix(args) 
            prefix = get_drl_prefix(args)
            if args.test: 
                
                run_iqn_vae(args, prefix, config)
            else: 
                run_iqn(args, prefix, config)

    except Exception as e:
        print(e)
