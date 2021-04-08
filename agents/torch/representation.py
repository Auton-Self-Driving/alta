import argparse
import os
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

# from leaderboard.utils.statistics_manager import StatisticsManager

from omegaconf import DictConfig, OmegaConf
import hydra

from data_modules import RepLearningDataModule
from agents.torch.representation import IA
from environment.carla_9_4.env import CarlaEnv


@hydra.main(config_path='conf', config_name='representation.yaml')
def main(cfg):
    # For reproducibility
    seed_everything(cfg.seed)

    # Loading agent and environment
    agent = hydra.utils.instantiate(cfg.algo.agent)

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    data_module = RepLearningDataModule(cfg.data_module)
    data_module.setup(None)

    # Offline training
    trainer = pl.Trainer(**cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.num_epochs)
    trainer.fit(agent, data_module)


if __name__ == '__main__':
    main()
