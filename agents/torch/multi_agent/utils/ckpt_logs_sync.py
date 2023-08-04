import shutil, errno, os, argparse

def copyanything(src, dst):
    # shutil.copytree(src, dst)
    dst_els = os.listdir(dst)
    for el in os.listdir(src):
        if el in dst_els:
            print(f"Not copying {el}")
            continue
        try:
            shutil.copytree(os.path.join(src,el), os.path.join(dst,el))
        except OSError as exc:
            print(f"Not copying {el},{exc}")
    # try:
    # except OSError as exc: # python >2.5
    #     if exc.errno in (errno.ENOTDIR, errno.EINVAL):
    #         shutil.copy(src, dst)
    #     else: raise

repo_path = "/home/scratch/amanmehr/alta/agents/torch/multi_agent/"
nfs_path = "/zfsauton2/home/amanmehr/sync/"

folders = ["checkpoints","tensorboard_logs"]

parser = argparse.ArgumentParser(description="Files syncer")
parser.add_argument("-i","--sync_in", default=False, action="store_true")

args = parser.parse_args()


for f in folders:

    src = os.path.join(repo_path,f)
    dst = os.path.join(nfs_path,f)

    if args.sync_in:
        src,dst = dst,src 

    # print(src,dst)
    copyanything(src, dst)

    
        
        


