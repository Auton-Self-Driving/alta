import argparse
import sys, os
sys.path.append('./../../../')
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
from environment.carla_9_4.config import ConfigManager
from train_measurements_sac_run import run_sac
from train_measurements_ppo_run import run_ppo
from train_vae_ppo_run import run_ppo_vae
from test_pid import test_pid_method


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser to run all deep RL algorithms')
    parser.add_argument('--algo',dest='algo',type=str,required=True, help='Algo: PPO or SAC or PID_TUNE')
    parser.add_argument('--task',dest='task',type=str, default='self-driving', required=True, help='Task')
    parser.add_argument('--test', dest='test', action='store_true', help='Enable testing.')
    parser.add_argument('--test-trails', dest='test_trails', type=int, default=5, help='No of different test trials.')
    parser.add_argument('--city_name',dest='city_name',type=str, default='Town01', help='Carla Town.')
    parser.add_argument('--vae_model_path',dest='vae_model_path',type=str, default='/zfsauton2/home/vkadi/projects/alta/agents/tf/trained_models/ae_model.json', help='VAE Model path.')
    parser.add_argument('--agent_model_path',dest='agent_model_path',type=str, default=None, help='Agent Model path.')
    parser.add_argument('--input-type', dest='input_type', type=str, default='wp', help='Observation type: "wp", "wp_constant", "wp_noise" or "wp_vae"')
    parser.add_argument('--scenarios', dest='scenarios', type=str, default='navigation', help='CARLA Scenarios type: "straight", "curved", "navigation" or "dynamic_navigation"')
    parser.add_argument('--lr',dest='lr',type=float,default=3e-4)
    parser.add_argument('--ent-coef',dest='ent_coef',type=float,default=-1, help='Entropy term for PPO runs.')
    parser.add_argument('--buffer-size',dest='buffer_size',type=int,default=50000)
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=512)
    parser.add_argument('--gradient-steps-per-iteration',dest='gradient_steps_per_iteration',type=int,default=1)
    parser.add_argument('--target-update-interval',dest='target_update_interval',type=int,default=1)
    parser.add_argument('--run-id',dest='run_id',type=str, required=True, help='Unique identifier for the run. It is appended to log directory name.')
    parser.add_argument('--network',dest='network',type=str,default='1_layer', help='network: 1_layer, 2_layer, CustomPolicy1 or CustomPolicy2.')
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
    parser.add_argument('--const-collision-penalty',dest='const_collision_penalty',type=float,default=100.0, help='Constant penalty for collision.')
    parser.add_argument('--collision-penalty-speed-coeff',dest='collision_penalty_speed_coeff',type=float,default=100.0, help='Speed coefficient for speed-proportional collision penalty.')
    parser.add_argument('--enable-brake', dest='enable_brake', action='store_true', help='Whether to enable brake action')
    parser.add_argument('--fs',dest='frame_skip',type=int,default=1, help='Number of frame skip (default:1)')
    parser.add_argument('--n-steps',dest='n_steps',type=int,default=500, help='Number of steps in trajectory for PPO.')
    parser.add_argument('--disable-semantic', dest='disable_semantic', action='store_true', help='Whether to disable semantic segmentation camera and enable RGB camera. (semantic is enabled by default).')
    parser.add_argument('--disable-collision', dest='disable_collision', action='store_true', help='Whether to disable collision for episode done condition.')
    parser.add_argument('--enable-static', dest='enable_static', action='store_true', help='Whether to enable max static steps for episode done condition.')
    parser.add_argument('--static-steps',dest='static_steps',type=int,default=1000, help='Max no of static steps.')
    parser.add_argument('--ae-lr',dest='ae_lr',type=float,default=1e-4)
    parser.add_argument('--use-pid-fs',dest='use_pid_fs', action='store_true', help='Whether to use pid within each frameskip. (Right way to do it)')
    

    return parser.parse_args()
def main(args):

    args = parse_arguments()
    return args

def create_sac_prefix(args):

    base = 'algo_' + args.algo \
        + '_task_' + args.task \
        + '_input_' + args.input_type \
        + '_network_' + str(args.network) \
        + '_lr_' + str(args.lr)  \
        + '_buffer_' + str(args.buffer_size) \
        + '_batchsz_'+ str(args.batch_size) \
        + '_n-steps_'+ str(args.n_steps) \
        + '_gradupd-per-iter_'+ str(args.gradient_steps_per_iteration) \
        + '_tgt-upd-int_'+ str(args.target_update_interval) \
        + '_ent-coef_'+ str(args.ent_coef) \
        + '_cp-'+str(args.const_collision_penalty)+'-'+str(args.collision_penalty_speed_coeff)\
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

    if args.const_collision_penalty != 0:
        const_collision_penalty_str = '_col_' + str(args.const_collision_penalty)
    else:
        const_collision_penalty_str = ""
    
    if args.collision_penalty_speed_coeff != 0:
        collision_penalty_speed_coeff_str = '_col_sp_' + str(args.collision_penalty_speed_coeff)
    else:
        collision_penalty_speed_coeff_str = ""
    
    if args.enable_brake != False:
        enable_brake_str = '_brake'
    else:
        enable_brake_str = ''

    if args.disable_collision != False:
        disable_collision_str = "_disable_collision_"
    else:
        disable_collision_str = ''

    if args.enable_static != False:
        enable_static_str = "_enable_static_" + str(args.static_steps) + "_"
    else:
        enable_static_str = ''

    if args.ent_coef != 0.005:
        ent_coef_str = '_ent_' + str(args.ent_coef)
    else:
        ent_coef_str = ''
    
    if args.input_type == "wp_noise":
        input_type = args.input_type + str(args.noise_dim)
    else:
        input_type = args.input_type
    
    if args.frame_skip != 1:
        frame_skip_str = '_fs_' + str(args.frame_skip)
    else:
        frame_skip_str = ''
    
    if args.n_steps > 0:
        n_steps_str = '_n_' + str(args.n_steps)
    else:
        n_steps_str = ''
    
    if args.agent_model_path is not None:
        use_pretrained_agent_str = '_pretrained_agent_'
    else:
        use_pretrained_agent_str = ''

    if args.ae_lr != 1e-4:
        ae_lr_str = '_ae_lr_' + str(args.ae_lr)
    else:
        ae_lr_str = ''

    if args.use_pid_fs:
        use_pid_fs_str = '_use_pid_fs_'
    else:
        use_pid_fs_str = ''
        
    prefix = 'algo_' + args.algo \
        + '_task_' + args.task \
        + '_input_' + input_type \
        + '_network_' + str(args.network) \
        + '_lr_' + str(args.lr)  \
        + ae_lr_str \
        + '_' + args.scenarios \
        + use_pretrained_agent_str \
        + num_npc_str \
        + enable_brake_str \
        + disable_collision_str \
        + enable_static_str \
        + const_collision_penalty_str \
        + collision_penalty_speed_coeff_str \
        + ent_coef_str \
        + frame_skip_str \
        + use_pid_fs_str \
        + n_steps_str \
        + vae \
        + '_runid_' + args.run_id + '/'

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
    config.config["enable_brake"] = args.enable_brake
    config.config["frame_skip"] = args.frame_skip
    config.config["semantic"] = not args.disable_semantic
    config.config["disable_collision"] = args.disable_collision
    config.config["enable_static"] = args.enable_static
    config.config["max_static_steps"] = args.static_steps
    config.config["city_name"] = args.city_name
    config.config["testing"] = args.test
    config.config["use_pid_in_frame_skip"] = args.use_pid_fs
    config.config["batch_size"] = args.batch_size
    config.config["gradient_steps_per_iteration"] = args.gradient_steps_per_iteration
    config.config["target_update_interval"] = args.target_update_interval
    config.config["ent_coef"] = args.ent_coef
    config.config["task"] = args.task
    config.config["n_steps"] = args.n_steps

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
            elif args.input_type == "wp_vae":
                run_ppo_vae(args, prefix, config)
            else:
                print("specify correct input_type: wp, wp_vae")
                print("exiting")
        elif args.algo == "PID_TUNE":
            prefix = create_ppo_prefix(args)
            test_pid_method(args, prefix, config)
    except Exception as e:
        print(e)
