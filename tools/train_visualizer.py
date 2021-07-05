#!/usr/bin/env python3
""" Train cover generation models. """

import sys
import os
import logging
import argparse
import json
import git
import random
import warnings

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchvision

from aural_travels.train import visualizer


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

    os.makedirs(params['output_dir'], exist_ok=True)
    os.makedirs(params['encoding_dir'], exist_ok=True)
    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    image_repr = visualizer.create_image_repr(params)

    encodings = {split: visualizer.load_or_encode_images(params['soundcloud_data_dir'],
                                                         params['encoding_dir'],
                                                         params['num_workers'],
                                                         image_repr,
                                                         split=split)
                 for split in ['validation', 'test', 'training']}

    datasets = {split: visualizer.load_dataset(params, split, encodings[split])
                for split in ['validation', 'test', 'training']}

    model = visualizer.create_model(params, image_repr, datasets['training'])

    dataloaders = {'training': DataLoader(datasets['training'],
                                          batch_size=params['batch_size'],
                                          shuffle=True,
                                          num_workers=params['num_workers']),
                   'validation': DataLoader(datasets['validation'],
                                            batch_size=params['batch_size'],
                                            num_workers=params['num_workers']),
                   'test': DataLoader(datasets['test'],
                                      batch_size=params['batch_size'],
                                      num_workers=params['num_workers'])}

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    num_trainable = sum(p.numel() for p in params_to_update)

    logger.info(f'Model: {model}')
    logger.info(f'Number of trainable parameters: {num_trainable}')

    optimizer = AdamW(params_to_update, lr=params['lr'])

    visualizer.train(params, model, optimizer, dataloaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soundcloud_data_dir',
                        help='Directory of the SoundCloud dataset',
                        required=True)
    parser.add_argument('--dataset',
                        help='Which dataset to train on',
                        choices=['soundcloud'],
                        required=True)
    parser.add_argument('--num_workers',
                        help='Number of worker processes to use for loading data',
                        type=int,
                        default=64)
    parser.add_argument('--batch_size',
                        help='Batch size for training and validation',
                        type=int,
                        default=128)
    parser.add_argument('--gradient_accumulation',
                        help='Number of gradient accumulation steps',
                        type=int,
                        default=1)
    parser.add_argument('--num_epochs',
                        help='Number of training epochs',
                        type=int,
                        default=30)
    parser.add_argument('--hidden_size',
                        help='Transformer hidden size',
                        type=int,
                        default=256)
    parser.add_argument('--num_layers',
                        help='Transformer number of layers',
                        type=int,
                        default=8)
    parser.add_argument('--num_enc_layers',
                        help='Transformer number of encoder layers',
                        type=int,
                        default=1)
    parser.add_argument('--num_dec_layers',
                        help='Transformer number of encoder layers',
                        type=int,
                        default=2)
    parser.add_argument('--num_heads',
                        help='Transformer number of attention heads',
                        type=int,
                        default=1)
    parser.add_argument('--attention_dropout',
                        help='Dropout for attention layers',
                        type=float,
                        default=0.1)
    parser.add_argument('--ffnn_dropout',
                        help='Dropout for FFNN layers',
                        type=float,
                        default=0.1)
    parser.add_argument('--audio_emb_dropout',
                        help='Dropout for audio embedding output',
                        type=float,
                        default=0.1)
    parser.add_argument('--input_dropout',
                        help='Dropout for audio input',
                        type=float,
                        default=0.1)
    parser.add_argument('--lr',
                        help='Learning rate for the optimizer',
                        type=float,
                        default=0.001)
    parser.add_argument('--output_dir',
                        help='Directory to save model in',
                        required=True)
    parser.add_argument('--sample_secs',
                        help='Duration of the model audio input',
                        default=2.0,
                        type=float)
    parser.add_argument('--n_fft',
                        help='Number of samples to use for FFT',
                        default=2048,
                        type=int)
    parser.add_argument('--hop_length',
                        help='Hop length to use for FFT',
                        default=1024,
                        type=int)
    parser.add_argument('--save_steps',
                        help='Save last checkpoint after this many training steps',
                        default=100,
                        type=int)
    parser.add_argument('--eval_steps',
                        help='Evaluate on validation est after this many training steps',
                        default=30,
                        type=int)
    parser.add_argument('--encoding_dir',
                        help='Directory to cache encodings in',
                        required=True)
    parser.add_argument('--non_autoregressive',
                        help='Use non-autoregressive model',
                        action='store_true')
    parser.add_argument('--corrupt_image_mode',
                        help='Image corruption mode for non-autoregressive model',
                        default=None,
                        type=str)
    parser.add_argument('--seed',
                        help='Random seed',
                        default=42,
                        type=int)
    parser.add_argument('--toy_data',
                        help='Train on a small subset of the training data',
                        action='store_true')
    parser.add_argument('--expose_steps',
                        help='Expose the model to its own predictions in training',
                        default=None,
                        type=int)
    parser.add_argument('--expose_alpha',
                        help='Probability of exposing model to its own predictions',
                        default=0.5,
                        type=float)
    parser.add_argument('--contrastive_lambda',
                        help='Scaling factor for the contrastive loss',
                        type=float)
    parser.add_argument('--pull_lambda',
                        help='Scaling factor for the pull loss',
                        type=float)
    parser.add_argument('--push_lambda',
                        help='Scaling factor for the push loss',
                        type=float)
    parser.add_argument('--axial_attention',
                        help='Use axial attention',
                        action='store_true')
    parser.add_argument('--use_layer_scale',
                        help='Use layer scale op',
                        action='store_true')
    parser.add_argument('--global_features',
                        help='Global mel features',
                        action='store_true')
    parser.add_argument('--num_latents',
                        default=0,
                        type=int)
    parser.add_argument('--latent_size',
                        default=32,
                        type=int)
    parser.add_argument('--random_latents',
                        action='store_true')
    parser.add_argument('--image_repr',
                        help='Model to use for image representation',
                        choices=['dalle', 'vqgan'])
    parser.add_argument('--model',
                        choices=['audio_dalle', 'audio_dalle_nat', 'bottleneck_gen'])
    args = parser.parse_args()

    run(vars(args))
