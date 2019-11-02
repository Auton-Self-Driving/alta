import argparse
import sys, os

sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
from environment.carla_9_4.config import ConfigManager
from train_vae_ae_semantic import train_vae_ae
import traceback


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser to run all deep RL algorithms')
    parser.add_argument('--algo',dest='algo',type=str,required=True, help='Algo: VAE or AE')
    parser.add_argument('--lr',dest='lr',type=float,default=5e-3)
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=64)
    parser.add_argument('--run-id',dest='run_id',type=str, required=True, help='Unique identifier for the run. It is appended to log directory name.')
    parser.add_argument('--num-npc',dest='num_npc',type=int,default=60, help='number of other vehicles')
    parser.add_argument('--vae-zsize',dest='vae_zsize',type=int,default=512)
    parser.add_argument('--carla-gpu',dest='carla_gpu',type=str,default='0')
    parser.add_argument('--code-gpu',dest='code_gpu',type=str,default='0')
    parser.add_argument('--base-log-dir',dest='base_log_dir',type=str, required=True, help='base log directory, Eg: /zfsauton2/home/hiteshar/research/alta-logs/new_env/sac_runs1/')
    parser.add_argument('--timesteps',dest='timesteps',type=int,default=2000000, help='total timesteps to train')
    
    return parser.parse_args()
def main(args):

    args = parse_arguments()
    return args

def create_prefix(args):

    base = 'algo_' + args.algo  \
        + '_lr_' + str(args.lr) \
        + '_batchsize_' + str(args.batch_size) \
        + '_npc_' + str(args.num_npc) \
    base += '_zsize_' + str(args.vae_zsize)

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
        
        base_prefix, prefix = create_prefix(args)
        print("prefix", prefix)
        train_vae_ae(args, prefix, config)
    
    except Exception as e:
        print(e)
        error_file = os.path.join(args.base_log_dir,prefix, "error.txt")
        with open(error_file, "w") as f:
            print(prefix, e)
            traceback.print_exc()
            f.write(str(e))



    
    