# ALTA: Agents Learning To Act

Deep Reinforcement Learning for Autonomous Driving in Urban Environments

**Project Page:** https://sites.google.com/view/rl4ad/

This repository contains the code for research conducted at the **Carnegie Mellon University Robotics Institute** (Auton Lab, advised by Prof. Jeff Schneider) on applying deep reinforcement learning to learn urban driving policies in the CARLA simulator.

---

## Publications

If you use this code, please cite the relevant publications:

### Learning Urban Driving Policies using Deep Reinforcement Learning
**T Agarwal\*, H Arora\*, J Schneider** — IEEE ITSC 2021

```bibtex
@inproceedings{agarwal2021learning,
  title={Learning Urban Driving Policies using Deep Reinforcement Learning},
  author={Agarwal, Tanmay and Arora, Hitesh and Schneider, Jeff},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  year={2021},
  organization={IEEE}
}
```

### Affordance-based Reinforcement Learning for Urban Driving
**T Agarwal\*, H Arora\*, J Schneider** — arXiv 2021

```bibtex
@article{agarwal2021affordance,
  title={Affordance-based Reinforcement Learning for Urban Driving},
  author={Agarwal, Tanmay and Arora, Hitesh and Schneider, Jeff},
  journal={arXiv preprint arXiv:2101.05970},
  year={2021}
}
```

### Learning to Drive using Waypoints
**T Agarwal\*, H Arora\*, T Parhar\*, S Deshpande, J Schneider** — NeurIPS 2019 ML4AD Workshop

```bibtex
@inproceedings{agarwal2019learning,
  title={Learning to Drive using Waypoints},
  author={Agarwal, Tanmay and Arora, Hitesh and Parhar, Tushar and Deshpande, Shubhankar and Schneider, Jeff},
  booktitle={NeurIPS 2019 Machine Learning for Autonomous Driving Workshop},
  year={2019}
}
```

### Thesis Documents

- [On-Policy Reinforcement Learning for Learning to Drive in Urban Settings — Tanmay Agarwal, Jeff Schneider](https://www.ri.cmu.edu/publications/on-policy-reinforcement-learning-for-learning-to-drive-in-urban-settings/)
- [Off-Policy Reinforcement Learning for Autonomous Driving — Hitesh Arora, Jeff Schneider](https://www.ri.cmu.edu/publications/off-policy-reinforcement-learning-for-autonomous-driving/)

\* *indicates equal contribution*

---

## Overview

This project investigates deep RL approaches (PPO, DQN, SAC) for learning end-to-end driving policies in the CARLA urban driving simulator. Key contributions include:

- **Waypoint-based state representation** for RL agents using privileged planner information
- **Affordance-based RL** combining modular perception with deep RL for planning and control
- **On-policy (PPO) and off-policy (DQN) algorithms** evaluated on the NoCrash benchmark
- **Forward search** over a population of policies for improved navigation
- **VAE/Autoencoder-based image representations** for vision-based driving

---

## Repository Structure

```
alta/
├── agents/
│   ├── tf/                    # Main agent implementations (TensorFlow 1.10 / Stable Baselines)
│   │   ├── run_code.py        # Entry point for training and evaluation
│   │   ├── ppo.py             # PPO agent implementation
│   │   ├── custom_dqn.py     # Custom DQN implementation
│   │   ├── my_sac.py         # SAC agent implementation
│   │   ├── models.py         # Neural network architectures
│   │   ├── ae/               # Autoencoder for image representation
│   │   ├── vae/              # Variational autoencoder models
│   │   ├── run_scripts/      # Training scripts organized by algorithm
│   │   │   ├── ppo/thesis/   # Scripts to reproduce PPO thesis results
│   │   │   └── dqn/thesis/   # Scripts to reproduce DQN thesis results
│   │   └── trained_models/   # Pretrained model checkpoints
│   └── torch/                 # Legacy PyTorch implementations (DDPG/SAC/TD3)
├── environment/
│   └── carla_9_4/            # CARLA 0.9.6 environment wrapper
│       ├── env.py            # OpenAI Gym-compatible environment
│       ├── reward.py         # Reward function definitions
│       ├── scenarios.py      # Driving scenario configurations
│       ├── controller.py     # PID controller for low-level control
│       ├── planner.py        # Route planning utilities
│       └── sensors.py        # Camera and sensor configurations
├── archive/                   # Archived code (CARLA 0.8.2, MuJoCo experiments)
├── libs/                      # Pre-compiled image processing libraries
├── environment.yml            # Conda environment specification
├── setup.py                   # Package installation
└── LICENSE                    # MIT License
```

---

## Setup Instructions

### Prerequisites

- Linux (tested on Ubuntu 16.04/18.04)
- NVIDIA GPU with CUDA 9.0
- CARLA Simulator v0.9.6
- Anaconda/Miniconda

### 1. Install CARLA v0.9.6

```bash
mkdir $HOME/carla96
cd $HOME/carla96
wget "http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz"
tar xvzf CARLA_0.9.6.tar.gz
```

### 2. Clone this repository

```bash
mkdir $HOME/projects
cd $HOME/projects
git clone https://github.com/Auton-Self-Driving/alta.git
cd alta
```

### 3. Create conda environment

```bash
conda env create -f environment.yml
conda activate carla9.4_py35
```

### 4. Set environment variables

Add to your `~/.bashrc`:

```bash
export ALTA=$HOME/projects/alta
export LIBS=$ALTA/libs
export PATH=$LIBS/nasm/bin:$LIBS/libjpeg8/bin:$LIBS/libpng/bin:$LIBS/libjpeg/bin:$LIBS/libjpeglua/bin:$PATH
export LD_LIBRARY_PATH=$LIBS/libjpeg8/lib:$LIBS/libpng/lib:$LIBS/libjpeg/lib64:$LIBS/libjpeglua/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$CPLUS_INCLUDE_PATH
export CARLA_9_4_PATH=$HOME/carla96
```

### 5. Install the package

```bash
cd $ALTA
pip install -e .
```

---

## Usage

### Training

Run from `alta/agents/tf/`:

```bash
python run_code.py \
  --algo PPO \
  --input-type wp \
  --network 2_layer \
  --base-log-dir '../../../alta-logs/' \
  --scenarios navigation \
  --timesteps 1000000 \
  --n-steps 1000 \
  --carla-gpu 0 \
  --code-gpu 0 \
  --lr 2e-4 \
  --run-id 1
```

**Supported algorithms:** `PPO`, `DQN`, `SAC`

**Input types:** `wp` (waypoints), `wp_vae` (waypoints + VAE image encoding), `wp_noise` (waypoints with noise)

**Scenarios:** `straight`, `curved`, `navigation`, `dynamic_navigation`

### Reproducing Thesis Results

PPO experiments:
```bash
# See scripts in: agents/tf/run_scripts/ppo/thesis/
```

DQN experiments:
```bash
# See scripts in: agents/tf/run_scripts/dqn/thesis/
```

### Evaluation

```bash
python run_code.py \
  --algo PPO \
  --test \
  --agent_model_path /path/to/saved/model \
  --input-type wp \
  --scenarios dynamic_navigation \
  --run-id test_1 \
  --base-log-dir '../../../alta-logs/'
```

---

## Key Dependencies

| Package | Version |
|---------|---------|
| Python | 3.5.6 |
| TensorFlow (GPU) | 1.10.0 |
| PyTorch | 0.4.1 |
| Stable Baselines | 2.9.0 |
| OpenAI Gym | 0.12.1 |
| CARLA | 0.9.6 |
| CUDA | 9.0 |

---

## Authors

- **Tanmay Agarwal** — [tanmay.agrawal@hotmail.com](mailto:tanmay.agrawal@hotmail.com)
- **Hitesh Arora** — [hitesharora11@gmail.com](mailto:hitesharora11@gmail.com)
- **Shubhankar Deshpande**
- **Prof. Jeff Schneider** (Advisor) — Carnegie Mellon University, Robotics Institute

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work was conducted at the [Auton Lab](https://autonlab.org/), Robotics Institute, Carnegie Mellon University. We thank Prof. Jeff Schneider for his guidance and the CMU Robotics Institute for providing computational resources.
