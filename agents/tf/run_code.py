import argparse
import sys, os

sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
from environment.carla_9_4.config import ConfigManager
from train_measurements_sac_run import run_sac
from train_measurements_ppo_run import run_ppo
from train_vae_ppo_run import run_ppo_vae


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser to run all deep RL algorithms')
    parser.add_argument('--algo',dest='algo',type=str,required=True, help='Algo: PPO or SAC')
    parser.add_argument('--vae_model_path',dest='vae_model_path',type=str, default='/zfsauton2/home/hiteshar/research/alta/agents/tf/trained_models/ae_model.json', help='VAE Model path.')
    parser.add_argument('--input-type', dest='input_type', type=str, default='wp', help='Observation type: "wp", "wp_constant", "wp_noise" or "wp_vae"')
    parser.add_argument('--scenarios', dest='scenarios', type=str, default='navigation', help='CARLA Scenarios type: "straight", "curved", "navigation" or "dynamic_navigation"')
    parser.add_argument('--lr',dest='lr',type=float,default=3e-4)
    parser.add_argument('--ent-coef',dest='ent_coef',type=float,default=0.005, help='Entropy term for PPO runs.')
    parser.add_argument('--buffer-size',dest='buffer_size',type=int,default=50000)
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

    # prefix = 'algo_' + args.algo \
    #     + '_input_' + args.input_type \
    #     + '_network_' + str(args.network) \
    #     + '_lr_' + str(args.lr)  \
    #     + '_reduced_' + args.scenarios + '_5' \
    #     + '_finetunevae_' + str(args.finetune_vae) \
    #     + '_pretrainedvae_' + str(not args.train_vae) \
    #     + '_runid_' + args.run_id + '/'
    
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
    
    if args.input_type == "wp_noise":
        input_type = args.input_type + str(args.noise_dim)
    else:
        input_type = args.input_type
        
    prefix = 'algo_' + args.algo \
        + '_input_' + input_type \
        + '_network_' + str(args.network) \
        + '_lr_' + str(args.lr)  \
        + '_' + args.scenarios \
        + vae \
        + num_npc_str \
        + '_runid_' + args.run_id + '/'

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

    try:
        if args.algo == "SAC":
            base_prefix, prefix = create_sac_prefix(args)
            print("prefix", prefix)
            run_sac(args, prefix, base_prefix, config)
        elif args.algo == "PPO":
            prefix = create_ppo_prefix(args)
            print("prefix", prefix)
            if args.input_type == "wp" or args.input_type == "wp_noise":
                run_ppo(args, prefix, config)
            elif args.input_type == "wp_vae":
                run_ppo_vae(args, prefix, config)
            else:
                print("specify correct input_type: wp, wp_vae")
                print("exiting")
    except Exception as e:
        print(e)


    
    