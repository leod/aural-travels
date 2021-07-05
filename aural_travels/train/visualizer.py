import sys
import os
import logging
import json
import random
import warnings
from collections import defaultdict, Counter

from tqdm import tqdm
import git

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator

from aural_travels.model import audio_dalle, AudioDALLE, AudioDALLENAT, BottleneckGen
from aural_travels.model.image_repr import DALLEImageRepr, VQGANImageRepr
from aural_travels.data import soundcloud


# FIXME: Disable librosa warnings for missing PySoundFile.
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_image_repr(params):
    if params['image_repr'] == 'dalle':
        return DALLEImageRepr()
    elif params['image_repr'] == 'vqgan':
        return VQGANImageRepr()
    else:
        assert False


def create_model(params, image_repr, dataset):
    if params['model'] == 'audio_dalle_nat':
        # TODO: Untested after image_repr refactoring
        model = AudioDALLENAT(image_repr=image_repr,
                              audio_seq_len=dataset.num_samples(),
                              audio_num_features=dataset.num_features(),
                              hidden_size=params['hidden_size'],
                              num_layers=params['num_layers'],
                              num_heads=params['num_heads'],
                              attention_dropout=params['attention_dropout'],
                              ffnn_dropout=params['ffnn_dropout'],
                              axial_attention=params['axial_attention'])
    elif params['model'] == 'audio_dalle':
        model = AudioDALLE(image_repr=image_repr,
                           audio_seq_len=dataset.num_samples(),
                           audio_num_features=dataset.num_features(),
                           hidden_size=params['hidden_size'],
                           num_layers=params['num_layers'],
                           num_heads=params['num_heads'],
                           attention_dropout=params['attention_dropout'],
                           ffnn_dropout=params['ffnn_dropout'])
    elif params['model'] == 'bottleneck_gen':
        model = BottleneckGen(image_repr=image_repr,
                              audio_seq_len=dataset.num_samples(),
                              audio_num_features=dataset.num_features(),
                              hidden_size=params['hidden_size'],
                              num_enc_layers=params['num_enc_layers'],
                              num_dec_layers=params['num_dec_layers'],
                              num_heads=params['num_heads'],
                              attention_dropout=params['attention_dropout'],
                              ffnn_dropout=params['ffnn_dropout'],
                              audio_emb_dropout=params['audio_emb_dropout'],
                              input_dropout=params['input_dropout'],
                              use_layer_scale=params['use_layer_scale'],
                              num_latents=params['num_latents'],
                              latent_size=params['latent_size'],
                              random_latents=params['random_latents'])

    return model


def encode_images(soundcloud_data_dir, num_workers, image_repr, split):
    # We reuse the GenrePredictionDataset because it nicely loads the image data, but we don't care
    # about the genre here. 
    input_transform = lambda image: image_repr.image_to_tensor(image)[0]
    dataset = soundcloud.GenrePredictionDataset(soundcloud_data_dir,
                                                split=split,
                                                input_transform=input_transform)
    loader = DataLoader(dataset,
                        batch_size=64,
                        shuffle=False,
                        num_workers=num_workers)
    encoding = []

    image_repr = image_repr.to('cuda')

    with torch.no_grad():
        for tensor, _ in tqdm(loader):
            tensor = tensor.to('cuda')
            encoding.append(image_repr.encode(tensor).detach().cpu())

    return torch.cat(encoding)


def load_or_encode_images(soundcloud_data_dir, encoding_dir, num_workers, image_repr, split):
    encoding_path = os.path.join(encoding_dir, split + '.pt')

    if not os.path.exists(encoding_path):
        logger.info(f'Encoding dataset "{split}", will save at "{encoding_path}"')
        encoding = encode_images(soundcloud_data_dir=soundcloud_data_dir,
                                 num_workers=num_workers,
                                 image_repr=image_repr,
                                 split=split)
        torch.save(encoding, encoding_path)
    else:
        logger.info(f'Using precomputed encodings at "{encoding_path}"')
        encoding = torch.load(encoding_path)

    logger.info(f'Encoding shape: {encoding.size()}')
    return encoding


def load_dataset(params, split, encodings=None):
    return soundcloud.CoverGenerationDataset(data_dir=params['soundcloud_data_dir'],
                                             split=split,
                                             image_labels=encodings,
                                             sample_secs=params['sample_secs'],
                                             n_fft=params['n_fft'],
                                             hop_length=params['hop_length'],
                                             toy_data=params['toy_data'],
                                             audio_pairs=params['contrastive_lambda'] is not None,
                                             global_features=params['global_features'])


def save_checkpoint(model, optimizer, epoch, global_step, path):
    logger.info(f'Saving checkpoint (epoch={epoch}, global_step={global_step}) to "{path}"')
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step},
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


def calc_loss(params, losses, modes):
    if params['contrastive_lambda'] is not None and params['pull_lambda'] is not None:
        modes['generate'] += losses[0].item()
        modes['contrastive'] += losses[1].item()
        modes['pull'] += losses[2].item()

        generate_lambda = 1.0 - params['contrastive_lambda'] - params['pull_lambda']

        return generate_lambda * losses[0] + \
            params['contrastive_lambda'] * losses[1] + \
            params['pull_lambda'] * losses[2]
    else:
        modes['generate'] += losses[0].item()

        if params['num_latents'] > 0:
            modes['generate_amin'] += losses[1]
            modes['push'] += losses[2].item()

            return (1.0 - params['push_lambda']) * losses[0] + params['push_lambda'] * losses[2]
        else:
            return losses[0]


def evaluate(params, model, dataloader):
    model.eval()

    loss = 0
    loss_mode = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            losses = model(*batch)
            loss += calc_loss(params, losses, loss_mode)

    loss /= len(dataloader)
    for mode in loss_mode.keys():
        loss_mode[mode] /= len(dataloader)

    return loss, loss_mode


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
        step_loss_mode = defaultdict(float)

        for i, batch in tqdm(enumerate(dataloader_training)):
            losses = model(*batch)
            losses = [loss / params['gradient_accumulation'] for loss in losses]

            loss = calc_loss(params, losses, step_loss_mode)

            accelerator.backward(loss)

            step_loss += loss.item()

            if (i + 1) % params['gradient_accumulation'] == 0:
                optimizer.step()
                optimizer.zero_grad()

                logger.info(f'step {global_step}: loss: {step_loss}, '
                            f'loss by mode: {dict(step_loss_mode)}')

                writer.add_scalar('loss/train', step_loss, global_step)
                for mode, loss in step_loss_mode.items():
                    writer.add_scalar(f'loss/train/{mode}', loss, global_step)

                step_loss = 0.0
                step_loss_mode = defaultdict(float)
                global_step += 1

                if global_step % params['save_steps'] == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, checkpoint_path)

                if global_step % params['eval_steps'] == 0:
                    eval_loss, eval_loss_mode = evaluate(params, model, dataloader_validation)
                    logger.info(f'step {global_step}: validation loss: {eval_loss}, '
                                f'validation loss by mode: {dict(eval_loss_mode)}')

                    writer.add_scalar('loss/valid', eval_loss, global_step)
                    for mode, loss in eval_loss_mode.items():
                        writer.add_scalar(f'loss/valid/{mode}', loss, global_step)

    writer.flush()
    writer.close()
