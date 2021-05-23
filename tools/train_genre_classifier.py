#!/usr/bin/env python3
""" Train various image classifiers for predicting album genre from cover image. """

import sys
import os
import logging
import argparse
import json
import git

import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision

from aural_travels.model import image
from aural_travels.data import fma, soundcloud
from aural_travels.train import classifier

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(fma_data_dir,
              fma_subset,
              soundcloud_data_dir,
              dataset,
              num_workers,
              batch_size,
              input_size,
              weighted_data):
    train_transform = image.get_data_transform(input_size, 'training')
    val_transform = image.get_data_transform(input_size, 'validation')

    if dataset == 'fma':
        train_data = fma.GenrePredictionDataset(fma_data_dir,
                                                fma_subset,
                                                split='training',
                                                input_transform=train_transform)
        val_data = fma.GenrePredictionDataset(fma_data_dir,
                                              fma_subset,
                                              split='validation',
                                              input_transform=val_transform)
    else:
        train_data = soundcloud.GenrePredictionDataset(soundcloud_data_dir,
                                                       split='training',
                                                       input_transform=train_transform)
        val_data = soundcloud.GenrePredictionDataset(soundcloud_data_dir,
                                                     split='validation',
                                                     input_transform=val_transform)

    if weighted_data:
        sampler = WeightedRandomSampler(weights=train_data.example_weights,
                                        num_samples=len(train_data.example_weights),
                                        replacement=True)
        shuffle = False
    else:
        sampler = None 
        shuffle = True

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              sampler=sampler)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return {
        'training': train_loader,
        'validation': val_loader
    }


def create_optimizer(params_to_update,
                     lr,
                     momentum,
                     weight_decay,
                     optimizer):
    if optimizer == 'SGD':
        return SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        return AdamW(params_to_update, lr=lr, weight_decay=weight_decay)


def run(params, device):
    logger.info(f'Model name: {params["model_name"]}')

    dataloaders = load_data(fma_data_dir=params['fma_data_dir'],
                            fma_subset=params['fma_subset'],
                            soundcloud_data_dir=params['soundcloud_data_dir'],
                            dataset=params['dataset'],
                            num_workers=params['num_workers'],
                            batch_size=params['batch_size'],
                            input_size=image.INPUT_SIZE[params['model_name']],
                            weighted_data=params['weighted_data'])

    model = image.initialize_model(params['model_name'],
                                   dataloaders['training'].dataset.num_classes(),
                                   train_encoder=True,
                                   use_pretrained=True)

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    num_trainable = sum(p.numel() for p in params_to_update)
    logger.info(f'Number of trainable parameters: {num_trainable}')

    optimizer = create_optimizer(params_to_update,
                                 lr=params['lr'],
                                 momentum=params['momentum'],
                                 weight_decay=params['weight_decay'],
                                 optimizer=params['optimizer'])

    _, val_acc_history = classifier.train(model,
                                          dataloaders,
                                          optimizer,
                                          num_epochs=params['num_epochs'],
                                          weighted_loss=params['weighted_loss'],
                                          device=device)
    return val_acc_history


def run_all(params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    repo = git.Repo(os.path.dirname(sys.argv[0]), search_parent_directories=True)

    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'Torchvision version: {torchvision.__version__}')
    logger.info(f'aural-travels repo commit: {repo.head.object.hexsha}')
    logger.info(f'aural-travels repo dirty: {repo.is_dirty()}')
    logger.info(f'Device: {device}')
    logger.info(f'Params: {json.dumps(params, indent=4)}\n')

    model_top_val_accs = {}

    for model_name in ['resnet50', 'resnet18', 'alexnet', 'vgg', 'squeezenet', 'densenet']:
        top_val_accs = []
        for _ in range(params['num_runs']):
            val_acc_history = run({'model_name': model_name, **params}, device)
            top_val_accs.append(max(val_acc_history))

        model_top_val_accs[model_name] = top_val_accs

    logger.info('Top validation accuracies: ')
    for model_name, top_val_accs in model_top_val_accs.items():
        for acc in top_val_accs:
            logger.info(f'{model_name}: {acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fma_data_dir',
                        help='Directory of the FMA dataset')
    parser.add_argument('--fma_subset',
                        help='Subset of the FMA data to use',
                        choices=['small', 'medium', 'large'],
                        default='medium')
    parser.add_argument('--soundcloud_data_dir',
                        help='Directory of the SoundCloud dataset')
    parser.add_argument('--dataset',
                        help='Which dataset to train on',
                        choices=['fma', 'soundcloud'],
                        required=True)
    parser.add_argument('--num_workers',
                        help='Number of worker processes to use for loading data',
                        type=int,
                        default=32)
    parser.add_argument('--num_runs',
                        help='Number of runs to perform per model',
                        type=int,
                        default=1)
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
    parser.add_argument('--momentum',
                        help='Momentum for the SGD optimizer',
                        type=float,
                        default=0.9)
    parser.add_argument('--weight_decay',
                        help='Weight decay scaling factor',
                        type=float,
                        default=0.0)
    parser.add_argument('--optimizer',
                        help='Optimizer to use',
                        choices=['SGD', 'AdamW'],
                        default='SGD')
    parser.add_argument('--weighted_loss',
                        help='Apply weights based on inverse class-frequency to loss',
                        action='store_true')
    parser.add_argument('--weighted_data',
                        help='Apply weights based on inverse class-frequency to data sampling',
                        action='store_true')

    args = parser.parse_args()
    run_all(vars(args)) 
