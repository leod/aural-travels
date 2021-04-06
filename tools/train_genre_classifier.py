#!/usr/bin/env python3
""" Train various image classifiers for predicting album genre from cover image. """

import sys
import os
import logging
import argparse
import git

import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
import torchvision

from aural_travels.model import image_head
from aural_travels.data import fma
from aural_travels.train import classifier

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUM_CLASSES = 16

def load_data(data_dir, subset, num_workers, batch_size, input_size):
    train_transform = image_head.get_data_transform(input_size, 'training')
    val_transform = image_head.get_data_transform(input_size, 'validation')
    train_data = fma.GenrePredictionDataset(data_dir,
                                            subset,
                                            split='training',
                                            input_transform=train_transform)
    val_data = fma.GenrePredictionDataset(data_dir,
                                          subset,
                                          split='validation',
                                          input_transform=val_transform)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
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
                     optimizer):
    if optimizer == 'SGD':
        return SGD(params_to_update, lr=lr, momentum=momentum)
    elif optimizer == 'AdamW':
        return AdamW(params_to_update, lr=lr)

def run(data_dir,
        subset,
        num_workers,
        model_name,
        train_encoder,
        use_pretrained,
        batch_size,
        num_epochs,
        lr,
        momentum,
        optimizer,
        device):
    logger.info(f'Model name: {model_name}')
    model, input_size = image_head.initialize_model(model_name,
                                                    NUM_CLASSES,
                                                    train_encoder,
                                                    use_pretrained)

    dataloaders = load_data(data_dir, subset, num_workers, batch_size, input_size)

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    num_trainable = sum(p.numel() for p in params_to_update)
    logger.info(f'Number of trainable parameters: {num_trainable}')

    optimizer = create_optimizer(params_to_update, lr, momentum, optimizer)

    _, val_acc_history = classifier.train(model,
                                          dataloaders,
                                          optimizer,
                                          num_epochs,
                                          device)
    return val_acc_history

def run_all(data_dir,
            subset,
            num_workers,
            num_runs,
            batch_size,
            num_epochs,
            lr,
            momentum,
            optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo = git.Repo(os.path.dirname(sys.argv[0]), search_parent_directories=True)

    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'Torchvision version: {torchvision.__version__}')
    logger.info(f'aural-travels repo commit: {repo.head.object.hexsha}')
    logger.info(f'aural-travels repo dirty: {repo.is_dirty()}')
    logger.info(f'Device: {device}')
    logger.info('Params: \n'
                f'    batch_size={batch_size}\n'
                f'    num_epochs={num_epochs}\n'
                f'    lr={lr}\n'
                f'    momentum={momentum}\n'
                f'    optimizer={optimizer}')

    model_top_val_accs = {}

    for model_name in ['resnet50', 'resnet18', 'alexnet', 'vgg', 'squeezenet', 'densenet']:
        top_val_accs = []
        for _ in range(num_runs):
            val_acc_history = run(data_dir=data_dir,
                                  subset=subset,
                                  num_workers=num_workers,
                                  model_name=model_name,
                                  train_encoder=True,
                                  use_pretrained=True,
                                  batch_size=batch_size,
                                  num_epochs=num_epochs,
                                  lr=lr,
                                  momentum=momentum,
                                  optimizer=optimizer,
                                  device=device)
            top_val_accs.append(max(val_acc_history))

        model_top_val_accs[model_name] = top_val_accs

    logger.info('Top validation accuracies: ')
    for model_name, top_val_accs in model_top_val_accs.items():
        for acc in top_val_accs:
            logger.info(f'{model_name}: {acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',
                        help='Directory of the FMA dataset',
                        required=True)
    parser.add_argument('--num_workers',
                        help='Number of worker processes to use for loading data',
                        type=int,
                        default=32)
    parser.add_argument('--subset',
                        help='Subset of the data to use',
                        choices=['small', 'medium', 'large'],
                        default='medium')
    parser.add_argument('--num_runs',
                        help='Number of runs to perform per model',
                        type=int,
                        default=3)
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
    parser.add_argument('--optimizer',
                        help='Optimizer to use',
                        choices=['SGD', 'AdamW'],
                        default='SGD')

    args = parser.parse_args()
    run_all(**vars(args)) 
