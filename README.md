# alta
Agents Learning to Act

Below are the instructions to setup and run the code.

1. Change to home directory ($HOME).

Commands:

cd ~


2. Install CARLA v0.9.6 (https://carla.org/2019/07/12/release-0.9.6/) for which the binaries are available here: (http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz)

Commands: 

mkdir $HOME/carla96

cd $HOME/carla96

wget "http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz"

tar xvzf CARLA*


3. Install anaconda to setup CARLA environment.

Commands:

cd ~

wget "https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh"

bash Anaconda3*


4. Git clone alta repository and switch to 'master' branch. 

Commands:

mkdir $HOME/projects

cd $HOME/projects

git clone https://github.com/Auton-Self-Driving/alta.git

cd alta

git checkout master


5. Install conda environment 'carla9.4_py35' from environment.yml file.

Commands:

cd $HOME/projects/alta

conda env create -f environment.yml

conda activate carla9.4_py35


6a. Set the following paths in the bashrc file.

export ALTA=$HOME/projects/alta
export LIBS=$ALTA/libs
export PATH=$LIBS/nasm/bin:$LIBS/libjpeg8/bin:$LIBS/libpng/bin:$LIBS/libjpeg/bin:$LIBS/libjpeglua/bin:$PATH
export LD_LIBRARY_PATH=$LIBS/libjpeg8/lib:$LIBS/libpng/lib:$LIBS/libjpeg/lib64:$LIBS/libjpeglua/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$CPLUS_INCLUDE_PATH
export CARLA_9_4_PATH=$HOME/carla96
conda activate carla9.4_py35

(Note: We use variable CARLA_9_4_PATH here as well as in the code, but we actually run the CARLA v0.9.6 in the latest version.)

6b. Execute bashrc file if not already done or if conda environment 'carla9.4_py35' is not active. 

Commands:

source ~/.bashrc


7. Install alta repository as python package. 

Commands:

cd $ALTA

pip install -e .


8. Test installation on cluster. 

For Slurm Cluster, 

Commands:

srun --gres gpu:1 --pty $SHELL (Skip this if not using SLURM)

cd $ALTA/agents/tf

python run_code.py .......



For more details like thesis document and slides, kindly refer to our webpage (https://sites.google.com/view/rl4ad/)
