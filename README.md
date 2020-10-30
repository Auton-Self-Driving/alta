# Agents Learning to Act (alta)

Webpage: https://sites.google.com/view/rl4ad/

Thesis Documents:

* [On-Policy Reinforcement Learning for Learning to Drive in Urban Settings - Tanmay Agarwal, Jeff Schneider](https://www.ri.cmu.edu/publications/on-policy-reinforcement-learning-for-learning-to-drive-in-urban-settings/)

* [Off-Policy Reinforcement Learning for Autonomous Driving - Hitesh Arora, Jeff Schneider](https://www.ri.cmu.edu/publications/off-policy-reinforcement-learning-for-autonomous-driving/)

Reproduce thesis results:

* Scripts directory to run thesis models (**PPO**: alta/agents/tf/run_scripts/ppo/thesis, **DQN**: alta/agents/tf/run_scripts/dqn/thesis)

* Thesis saved models directory on **Auton Cluster**: /zfsauton/datasets/ArgoRL


## Setup Instructions

## Carla 9.6
Below are the instructions to setup and run the code.

* Change to home directory ($HOME).

```
cd ~
```

* Install CARLA v0.9.6 (https://carla.org/2019/07/12/release-0.9.6/) for which the binaries are available here: (http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz)

```
mkdir $HOME/carla96
cd $HOME/carla96
wget "http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz"
tar xvzf CARLA*
```

* Install anaconda to setup CARLA environment.

```
cd ~
wget "https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh"
bash Anaconda3*
```

* Git clone alta repository and switch to 'master' branch.

```
mkdir $HOME/projects
cd $HOME/projects
git clone https://github.com/Auton-Self-Driving/alta.git
cd alta
git checkout master
```

* Install conda environment 'carla9.4_py35' from environment.yml file.

```
cd $HOME/projects/alta
conda env create -f environment.yml
conda activate carla9.4_py35
```

* Set the following paths in the bashrc file.
(Note: We use variable CARLA_9_4_PATH here as well as in the code, but we actually run the CARLA v0.9.6 in the latest version.)

```
export ALTA=$HOME/projects/alta
export LIBS=$ALTA/libs
export PATH=$LIBS/nasm/bin:$LIBS/libjpeg8/bin:$LIBS/libpng/bin:$LIBS/libjpeg/bin:$LIBS/libjpeglua/bin:$PATH
export LD_LIBRARY_PATH=$LIBS/libjpeg8/lib:$LIBS/libpng/lib:$LIBS/libjpeg/lib64:$LIBS/libjpeglua/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$CPLUS_INCLUDE_PATH
export CARLA_9_4_PATH=$HOME/carla96
conda activate carla9.4_py35
```

* Execute bashrc file if not already done or if conda environment 'carla9.4_py35' is not active.

```
source ~/.bashrc
```

* Install alta repository as python package.

```
cd $ALTA
pip install -e .
```

## Carla 9.10
Below are the instructions to setup and run the code.

* Change to home directory ($HOME).

```
cd ~
```

* Install CARLA v0.9.6 (https://carla.org/2020/09/25/release-0.9.10/) for which the binaries are available here: (https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz)

```
mkdir $HOME/carla910
cd $HOME/carla910
wget "https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz"
tar xvzf CARLA_0.9.10.1.tar.gz
```

* Install anaconda to setup CARLA environment.

```
cd ~
wget "https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh"
bash Anaconda3*
```

* Git clone alta repository and switch to 'master' branch.

```
mkdir $HOME/projects
cd $HOME/projects
git clone https://github.com/Auton-Self-Driving/alta.git
cd alta
git checkout master
```

* Install conda environment 'carla9.4_py35' from environment.yml file.

```
cd $HOME/projects/alta
conda env create -f environment.yml
conda activate carla9.10_py37
```

* Set the following paths in the bashrc file.
(Note: We use variable CARLA_9_4_PATH here as well as in the code, but we actually run the CARLA v0.9.6 in the latest version.)

```
export ALTA=$HOME/projects/alta
export LIBS=$ALTA/libs
export PATH=$LIBS/nasm/bin:$LIBS/libjpeg8/bin:$LIBS/libpng/bin:$LIBS/libjpeg/bin:$LIBS/libjpeglua/bin:$PATH
export LD_LIBRARY_PATH=$LIBS/libjpeg8/lib:$LIBS/libpng/lib:$LIBS/libjpeg/lib64:$LIBS/libjpeglua/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$CPLUS_INCLUDE_PATH
export CARLA_9_4_PATH=$HOME/carla96
conda activate carla9.4_py35
```

* Execute bashrc file if not already done or if conda environment 'carla9.4_py35' is not active.

```
source ~/.bashrc
```

* Install alta repository as python package.

```
cd $ALTA
pip install -e .
```


* Test installation on cluster.

```
srun --gres gpu:1 --pty $SHELL (Skip this if not using SLURM)
cd $ALTA/agents/tf
python run_code.py --algo PPO --input-type wp --network 2_layer --base-log-dir '../../../alta-logs/' --scenarios navigation --timesteps 1000000 --n-steps 1000 --carla-gpu 0 --code-gpu 0 --lr 2e-4 --run-id 1 &
````

## Contact

If you have any questions, problems, suggestions for improvement, please feel free to reach us via email (tanmay.agrawal@hotmail.com / hitesharora11@gmail.com).
