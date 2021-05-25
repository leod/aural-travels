#!/usr/bin/env python3
""" Train cover generation models. """

from re import L
import sys
import os
import logging
import argparse
import json
import git
import random
import warnings

from tqdm import tqdm

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from accelerate import Accelerator
from dalle_pytorch import OpenAIDiscreteVAE

from aural_travels.model import audio_dalle, AudioDALLE
from aural_travels.data import soundcloud

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# FIXME: Disable librosa warnings for missing PySoundFile.
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def encode_images(soundcloud_data_dir, num_workers, vae, split):
    # We reuse the GenrePredictionDataset because it nicely loads the image data, but we don't care
    # about the genre here. 
    input_transform = lambda image: audio_dalle.transform_image(vae.image_size, image)
    dataset = soundcloud.GenrePredictionDataset(soundcloud_data_dir,
                                                split=split,
                                                input_transform=input_transform)
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


def save_checkpoint(model, optimizer, epoch, global_step, path):
    logger.info(f'Saving checkpoint (epoch={epoch}, global_step={global_step}) to "{path}"')
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
               },
               path)


def load_checkpoint(model, optimizer, path=None):
    if path is None or not os.path.exists(path):
        logger.info('Starting training from the start')
        return model, optimizer, 0, 0

    logger.info(f'Restoring training state from "{path}"')
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']

    return model, optimizer, epoch, global_step


def evaluate(model, dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            loss += model(*batch).item()
    return loss / len(dataloader)


def train(params, model, optimizer, dataloaders):
    accelerator = Accelerator()

    writer = SummaryWriter(params['output_dir'])

    model, optimizer, dataloader_training, dataloader_validation = \
        accelerator.prepare(model, optimizer, dataloaders['training'], dataloaders['validation'])

    checkpoint_path = os.path.join(params['output_dir'], 'last_checkpoint.pt')
    model, optimizer, epoch, global_step = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(epoch, params['num_epochs']):
        model.train()

        logger.info(f'Starting epoch {epoch}')
        step_loss = 0.0

        for i, batch in tqdm(enumerate(dataloader_training)):
            loss = model(*batch)
            loss = loss / params['gradient_accumulation']
            accelerator.backward(loss)
            step_loss += loss.item()

            #logger.info('.')

            if (i + 1) % params['gradient_accumulation'] == 0:
                optimizer.step()
                optimizer.zero_grad()

                logger.info(f'step {global_step}: loss: {step_loss}')
                writer.add_scalar('loss/train', step_loss, global_step)

                step_loss = 0
                global_step += 1

                if global_step % params['save_steps'] == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path)

                if global_step % params['eval_steps'] == 0:
                    eval_loss = evaluate(model, dataloader_validation)
                    logger.info(f'step {global_step}: validation loss: {eval_loss}')
                    writer.add_scalar('loss/valid', eval_loss, global_step)

    writer.flush()
    writer.close()


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
                                                 sample_secs=params['sample_secs'],
                                                 n_fft=params['n_fft'],
                                                 hop_length=params['hop_length'],
                                                 toy_data=params['toy_data'])
        for split in ['validation', 'test', 'training']
    }

    model = AudioDALLE(vae=vae,
                       audio_seq_len=datasets['training'].num_samples(),
                       audio_num_features=datasets['training'].num_features(),
                       hidden_size=params['hidden_size'],
                       num_layers=params['num_layers'],
                       num_heads=params['num_heads'],
                       attention_dropout=params['attention_dropout'],
                       ffnn_dropout=params['ffnn_dropout'])

    dataloaders = {
        'training': DataLoader(datasets['training'],
                               batch_size=params['batch_size'],
                               shuffle=True,
                               num_workers=params['num_workers']),
        'validation': DataLoader(datasets['validation'],
                                 batch_size=params['batch_size'],
                                 num_workers=params['num_workers']),
        'test': DataLoader(datasets['test'],
                           batch_size=params['batch_size'],
                           num_workers=params['num_workers']),
    }

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    num_trainable = sum(p.numel() for p in params_to_update)

    logger.info(f'Model: {model}')
    logger.info(f'Number of trainable parameters: {num_trainable}')

    optimizer = AdamW(params_to_update, lr=params['lr'])

    train(params, model, optimizer, dataloaders)



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
    parser.add_argument('--num_heads',
                        help='Transformer number of attention heads',
                        type=int,
                        default=256)
    parser.add_argument('--attention_dropout',
                        help='Dropout for attention layers',
                        type=float,
                        default=0.1)
    parser.add_argument('--ffnn_dropout',
                        help='Dropout for FFNN layers',
                        type=float,
                        default=0.01)
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
    parser.add_argument('--seed',
                        help='Random seed',
                        default=42,
                        type=int)
    parser.add_argument('--toy_data',
                        help='Train on a small subset of the training data',
                        action='store_true')
    args = parser.parse_args()

    run(vars(args))
