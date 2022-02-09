# Agents Learning to Act (alta)

Webpage: https://sites.google.com/view/rl4ad/

Thesis Documents:

* [On-Policy Reinforcement Learning for Learning to Drive in Urban Settings - Tanmay Agarwal, Jeff Schneider](https://www.ri.cmu.edu/publications/on-policy-reinforcement-learning-for-learning-to-drive-in-urban-settings/)

* [Off-Policy Reinforcement Learning for Autonomous Driving - Hitesh Arora, Jeff Schneider](https://www.ri.cmu.edu/publications/off-policy-reinforcement-learning-for-autonomous-driving/)

Reproduce thesis results:

* Scripts directory to run thesis models (**PPO**: alta/agents/tf/run_scripts/ppo/thesis, **DQN**: alta/agents/tf/run_scripts/dqn/thesis)

* Thesis saved models directory on **Auton Cluster**: /zfsauton/datasets/ArgoRL


## Setup Instructions

## Traffic Light Detection

### Setup  

**Important** If you want to use the traffic light detector, please build AdelaiDet & detectron2 module by:  
```
cd <whatever-dir>/alta
python -m pip install -e detectron2
cd ../AdelaiDet
python setup.py build develop
```  

Please make sure that the gcc version you are using is >= 5.0.  
If errors happen and you want to rebuild any of them, please purge the build folder before rebuilding.  

### Train Your Own Detector

The basic steps to train your own detector are:  
1. Collecting semantic and RGB images.  
2. Create a COCO-formatted detection dataset.  
3. Start training via AdelaiDet.  

#### Collect Your Own Data

For each frame, we need an RGB image as the input of the detector and a semantic sensor output for annotating the groundtruth bboxes in step 2. There is really no standard way to do this. The general idea is to put a semantic sensor and an RGB sensor with the same resolution at the same spot. Let the vehicle move around and give data readings of those sensors. Then we directly save the sensor outputs to disk. You can do it yourself or use [environment/tools/save_images.py](environment/tools/save_images.py)).  


#### Create Your Own Dataset

Now since we have RGB image and semantic data, we need to annotate the training and testing data. Since detectron2 and AdelaiDet supports COCO dataset very well, We want to format our dataset into COCO-format and use their API to do the training. COCO format is documented at [COCO Official Docs](https://cocodataset.org/#format-data).  

There are many ways to do that, one possible way is using the code that is shared at [AdelaiDet/datasets/create_traffic_light_dataset.py](https://github.com/aim-uofa/AdelaiDet/blob/db238dafcacfb2e4f2bbd227d725e33fb3eb9bad/datasets/create_traffic_light_dataset.py). Just properly configure corresponding paths and namings and run  
```
cd AdelaiDet/datasets
python create_traffic_light_dataset.py
```
This script will create **one** groundtruth bbox for traffic light per frame, and multiple bboxes for other objects per frame specified in ''OTHER_LABELS`` in the header part of the script. It should work for many situations after properly configured.  


#### Training via AdelaiDet

To train a model with "tools/train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md), which has already been done in [train_net.py](https://github.com/aim-uofa/AdelaiDet/blob/db238dafcacfb2e4f2bbd227d725e33fb3eb9bad/tools/train_net.py#L240). You might need to change it based on how you name your dataset though.  

Then run:

```
cd AdelaiDet
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/FCOS-Detection/MS_DLA_34_4x_syncbn.yaml \
    --num-gpus 1 \
```
To evaluate the model after training, run:

```
cd AdelaiDet
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/FCOS-Detection/MS_DLA_34_4x_syncbn.yaml \
    --eval-only \
    --num-gpus 1 \
    MODEL.WEIGHTS output/fcos/FCOS_RT_MS_DLA_34_4x_traffic/model_final.pth
```
Note that:
- **Before training, check the config file to have the proper settings of NUM_CLASSES and learning rate (BASE_LR).** For example, ''MS_DLA_34_4x_syncbn.yaml`` is pre-configured to train on detecting 4 classes and to have an initial learning rate of 0.001
- The configs are made for 1 GPU training. To train on another number of GPUs, change the `--num-gpus`.
- If you want to measure the inference time, please change `--num-gpus` to 1.
- `OMP_NUM_THREADS=1` is set by default, please change it as needed.  
  
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

* Install CARLA v0.9.10 (https://carla.org/2020/09/25/release-0.9.10/) for which the binaries are available here: (https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz)

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
git clone https://github.com/Auton-Self-Driving/alta.git --recursive
cd alta
git checkout multiagent-ae
```

* Install conda environment 'carla9.4_py37' from environment.yml file.

```
cd $HOME/projects/alta
conda env create -f environment.yml
conda activate carla9.10_py37
```
  
* Build detectron2 and AdelaiDet

```
cd $HOME/projects/alta/
git submodule init
git submodule update
cd leaderboard
git remote set-url origin git@github.com:Auton-Self-Driving/leaderboard.git
git pull
git checkout dev-zheh
cd ../AdelaiDet
git remote set-url origin git@github.com:kareido/AdelaiDet.git
git pull
git checkout traffic_light
cd ..

# commented out
# python -m pip install -e detectron2
# cd AdelaiDet
# python setup.py build develop
``` 

* Set the following paths in the bashrc file.
(Note: We use variable CARLA_9_4_PATH here as well as in the code, but we actually run the CARLA v0.9.10 in the latest version.)

The setup file `configure_env.setup` contains all of the environment variables to necessary to setup the environment. Confim that these variables match your configuration (Specifically confirm `ALTA` and `CARLA_9_4_PATH`)

```
export ALTA=$HOME/projects/alta
export LIBS=$ALTA/libs
export PATH=$LIBS/nasm/bin:$LIBS/libjpeg8/bin:$LIBS/libpng/bin:$LIBS/libjpeg/bin:$LIBS/libjpeglua/bin:$PATH
export LD_LIBRARY_PATH=$LIBS/libjpeg8/lib:$LIBS/libpng/lib:$LIBS/libjpeg/lib64:$LIBS/libjpeglua/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$CPLUS_INCLUDE_PATH
export CARLA_9_4_PATH=$HOME/carla910
export SDL_AUDIODRIVER='dsp'
conda activate carla9.10_py37
```

* Execute this file by running
```
source configure_env.setup
```

To match the previous configuration steps, you can also add this the contents of the setup file to your `~/.bashrc` file.

* Install alta repository as python package.

```
cd $ALTA
pip install -e .
```

## For both environments

* Test installation on cluster.

```
srun --gres gpu:1 --pty $SHELL (Skip this if not using SLURM)
cd $ALTA/agents/tf
python run_code.py --algo PPO --input-type wp --network 2_layer --base-log-dir '../../../alta-logs/' --scenarios navigation --timesteps 1000000 --n-steps 1000 --carla-gpu 0 --code-gpu 0 --lr 2e-4 --run-id 1 &
````

## Contact

If you have any questions, problems, suggestions for improvement, please feel free to reach us via email (tanmay.agrawal@hotmail.com / hitesharora11@gmail.com).
