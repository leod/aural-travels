#!/usr/bin/env python3
""" Train cover generation models. """

import sys
import os
import logging
import argparse
import json
import git
import random

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision

from aural_travels.model import image
from aural_travels.data import soundcloud

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo = git.Repo(os.path.dirname(sys.argv[0]), search_parent_directories=True)

    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'Torchvision version: {torchvision.__version__}')
    logger.info(f'aural-travels repo commit: {repo.head.object.hexsha}')
    logger.info(f'aural-travels repo dirty: {repo.is_dirty()}')
    logger.info(f'Device: {device}')
    logger.info(f'Params: {json.dumps(params, indent=4)}\n')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',
                        help='Random seed',
                        default=42,
                        type=int)
    args = parser.parse_args()

    run(vars(args))