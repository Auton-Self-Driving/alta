import argparse
import sys, os

sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
from environment.carla_9_4.config import ConfigManager
from train_vae_ae_semantic import train_vae_ae
from train_vae import train_vae
import traceback


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser to run all deep RL algorithms')
    parser.add_argument('--algo',dest='algo',type=str,required=True, help='Algo: VAE or AE')
    parser.add_argument('--mode', dest='mode', action='store_false', help='Whether to train with or without simulator (True = train with simulator)')
    parser.add_argument('--model_path',dest='model_path',type=str, help='VAE Model path (Example: "/zfsauton2/home/tanmaya/projects/alta/agents/tf/trained_models/ae_model.json"')
    parser.add_argument('--lr',dest='lr',type=float,default=5e-3)
    parser.add_argument('--fmt', dest='fmt', type=str, default='.npz', help='Image input format (either ".npy" or ".npz"')
    parser.add_argument('--fstack',dest='frame_stack',type=int,default=1, help='Input frame stack size (default:1)')
    parser.add_argument('--epochs',dest='epochs',type=int,default=100)
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=64)
    parser.add_argument('--run-id',dest='run_id',type=str, required=True, help='Unique identifier for the run. It is appended to log directory name.')
    parser.add_argument('--num-npc',dest='num_npc',type=int,default=60, help='number of other vehicles')
    parser.add_argument('--vae-zsize',dest='vae_zsize',type=int,default=512)
    parser.add_argument('--carla-gpu',dest='carla_gpu',type=str,default='0')
    parser.add_argument('--code-gpu',dest='code_gpu',type=str,default='0')
    parser.add_argument('--new-data-split',dest='new_data_split', action='store_true', help='Generate a new train-val data split.')
    parser.add_argument('--base-log-dir',dest='base_log_dir',type=str, required=True, help='base log directory, Eg: /zfsauton2/home/tanmaya/projects/alta-logs/new_env/sac_runs1/')
    parser.add_argument('--data-dir',dest='data_dir',type=str, required=True, help='Data directory, Eg: /zfsauton2/home/tanmaya/projects/alta-logs/new_env/sac_runs1/')
    parser.add_argument('--timesteps',dest='timesteps',type=int,default=2000000, help='total timesteps to train')
    parser.add_argument('--sample-size',dest='sample_size',type=int,default=100000, help='No of samples from each dataset')
    
    return parser.parse_args()
def main(args):

    args = parse_arguments()
    return args

def create_prefix(args):

    base = 'algo_' + args.algo  \
        + '_lr_' + str(args.lr) \
        + '_batchsize_' + str(args.batch_size) \
        + '_npc_' + str(args.num_npc)
    base += '_zsize_' + str(args.vae_zsize)

    prefix = base + '_runid_' + args.run_id +'/'
    base_prefix = base +'/'
    return base_prefix, prefix

def create_prefix2(args):
    base = 'algo_' + args.algo  \
        + '_lr_' + str(args.lr) \
        + '_batchsize_' + str(args.batch_size) \
        + '_fs_' + str(args.frame_stack) \
        + '_epochs_' + str(args.epochs) \

    prefix = base + '_runid_' + args.run_id +'/'
    base_prefix = base +'/'
    return base_prefix, prefix


if __name__ == '__main__':
    args = main(sys.argv)
    print("args", args)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.code_gpu)

    config = ConfigManager(algo='AE')
    config.config["carla_gpu"] = str(args.carla_gpu)
    config.config["num_npc"] = args.num_npc

    try:
        if args.mode:
            base_prefix, prefix = create_prefix(args)
            print("prefix", prefix)
            train_vae_ae(args, prefix, config)
        else:
            base_prefix, prefix = create_prefix2(args)
            print("prefix", prefix)
            info = train_vae(args, prefix, config)
    
    except Exception as e:
        print(e)
        error_file = os.path.join(args.base_log_dir,prefix, "error.txt")
        with open(error_file, "w") as f:
            print(prefix, e)
            traceback.print_exc()
            f.write(str(e))



    
    