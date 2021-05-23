#!/usr/bin/env python3
""" Train cover generation models. """

import sys
import os
import logging
import argparse
import json
import git
import random

from tqdm import tqdm

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from dalle_pytorch import OpenAIDiscreteVAE

from aural_travels.model import image
from aural_travels.data import soundcloud

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def encode_images(soundcloud_data_dir, num_workers, vae, split):
    def transform(image):
        # As here:
        # https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
        s = min(image.size)
        
        if s < vae.image_size:
            raise ValueError(f'min dim for image {s} < {vae.image_size}')
            
        r = vae.image_size / s
        s = (round(r * image.size[1]), round(r * image.size[0]))
        image = TF.resize(image, s, interpolation=TF.InterpolationMode.LANCZOS)
        image = TF.center_crop(image, output_size=2 * [vae.image_size])
        image = transforms.ToTensor()(image)

        # NOTE: Leaving out map_pixels here, since it is handled by dalle_pytorch.
        return image

    # We reuse the GenrePredictionDataset because it nicely loads the image data, but we don't care
    # about the genre here. 
    dataset = soundcloud.GenrePredictionDataset(soundcloud_data_dir,
                                                split=split,
                                                input_transform=transform)
    loader = DataLoader(dataset,
                        batch_size=64,
                        shuffle=False,
                        num_workers=num_workers)
    encoding = []

    vae = vae.to('cuda')

    with torch.no_grad():
        for image, _ in tqdm(loader):
            image = image.to('cuda')
            encoding.append(vae.get_codebook_indices(image).detach().cpu())

    return torch.cat(encoding)


def load_or_encode_images(soundcloud_data_dir, encoding_dir, num_workers, vae, split):
    encoding_path = os.path.join(encoding_dir, split + '.pt')

    if not os.path.exists(encoding_path):
        logger.info(f'Encoding dataset "{split}", will save at "{encoding_path}"')
        encoding = encode_images(soundcloud_data_dir=soundcloud_data_dir,
                                 num_workers=num_workers,
                                 vae=vae,
                                 split=split)
        torch.save(encoding, encoding_path)
    else:
        logger.info(f'Using precomputed encodings at "{encoding_path}"')
        encoding = torch.load(encoding_path)

    logger.info(f'Encoding shape: {encoding.size()}')
    return encoding
    

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
        json.dump(params, f)

    vae = OpenAIDiscreteVAE()
    vae.eval()

    encodings = {
        split: load_or_encode_images(params['soundcloud_data_dir'],
                                     params['encoding_dir'],
                                     params['num_workers'],
                                     vae,
                                     split=split)
        for split in ['validation', 'test', 'training']
    }

    datasets = {
        split: soundcloud.CoverGenerationDataset(data_dir=params['soundcloud_data_dir'],
                                                 split=split,
                                                 image_labels=encodings[split],
                                                 sample_secs=params['sample_secs'])
        for split in ['validation', 'test', 'training']
    }


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
                        default=32)
    parser.add_argument('--batch_size',
                        help='Batch size for training and validation',
                        type=int,
                        default=128)
    parser.add_argument('--num_epochs',
                        help='Number of training epochs',
                        type=int,
                        default=10)
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
    parser.add_argument('--encoding_dir',
                        help='Directory to cache encodings in',
                        required=True)
    parser.add_argument('--seed',
                        help='Random seed',
                        default=42,
                        type=int)
    args = parser.parse_args()

    run(vars(args))