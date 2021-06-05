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
from dalle_pytorch import OpenAIDiscreteVAE

from aural_travels.model import audio_dalle, AudioDALLE, AudioDALLENAT
from aural_travels.data import soundcloud


# FIXME: Disable librosa warnings for missing PySoundFile.
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_vae(params):
    vae = OpenAIDiscreteVAE()
    vae.eval()
    return vae


def create_model(vae, dataset, params):
    if params['non_autoregressive']:
        model = AudioDALLENAT(vae=vae,
                              audio_seq_len=dataset.num_samples(),
                              audio_num_features=dataset.num_features(),
                              hidden_size=params['hidden_size'],
                              num_layers=params['num_layers'],
                              num_heads=params['num_heads'],
                              attention_dropout=params['attention_dropout'],
                              ffnn_dropout=params['ffnn_dropout'])
    else:
        model = AudioDALLE(vae=vae,
                           audio_seq_len=dataset.num_samples(),
                           audio_num_features=dataset.num_features(),
                           hidden_size=params['hidden_size'],
                           num_layers=params['num_layers'],
                           num_heads=params['num_heads'],
                           attention_dropout=params['attention_dropout'],
                           ffnn_dropout=params['ffnn_dropout'])

    return model


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


def load_dataset(params, split, encodings=None):
    return soundcloud.CoverGenerationDataset(data_dir=params['soundcloud_data_dir'],
                                             split=split,
                                             image_labels=encodings,
                                             sample_secs=params['sample_secs'],
                                             n_fft=params['n_fft'],
                                             hop_length=params['hop_length'],
                                             toy_data=params['toy_data'])


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


def corrupt_image_seq(mode, vocab_size, image_seq):
    seq_len = image_seq.shape[0]

    if mode == 'uniform':
        k = random.randint(0, seq_len)
        idxs = random.sample(list(range(seq_len)), k=k)

        for idx in idxs:
            image_seq[idx] = random.randint(0, vocab_size-1)
    elif mode == 'full':
        for idx in range(seq_len):
            image_seq[idx] = random.randint(0, vocab_size-1)
    elif mode == 'uniform_and_full':
        p = 0.2
        if random.random() < p:
            return corrupt_image_seq('full', vocab_size, image_seq)
        else:
            return corrupt_image_seq('uniform', vocab_size, image_seq)
    else:
        assert False, f'unknown image corruption mode: {mode}'

    return image_seq


def corrupt_image_seq_batch(mode, vocab_size, image_seqs):
    corrupt_image_seqs = image_seqs.clone()

    for i in range(image_seqs.shape[0]):
        corrupt_image_seq(mode, vocab_size, corrupt_image_seqs[i])

    return corrupt_image_seqs


def prepare_batch(params, model, batch):
    if params['corrupt_image_mode'] is not None:
        if params['expose_steps'] is None:
            batch.append(corrupt_image_seq_batch(params['corrupt_image_mode'],
                                                 model.vae.dec.vocab_size,
                                                 batch[1]))
            mode = 'corrupt'
        else:
            # FIXME: This is non-deterministic for validation/test set...

            parts = []

            if random.random() < params['expose_alpha']:
                last_image_seqs = corrupt_image_seq_batch('full',
                                                          model.vae.dec.vocab_size,
                                                          batch[1])

                for l in range(params['expose_steps']):
                    parts.append([batch[0], batch[1], last_image_seqs])
                    last_image_seqs = model.generate_image_seq(batch[0],
                                                               corrupt_image_seq=last_image_seqs)
                mode = 'expose'
            else:
                for l in range(params['expose_steps']):
                    corrupt_image_seqs = corrupt_image_seq_batch(params['corrupt_image_mode'],
                                                                 model.vae.dec.vocab_size,
                                                                 batch[1])
                    parts.append([batch[0], batch[1], corrupt_image_seqs])
                mode = 'corrupt'

            batch = [torch.cat([part[i] for part in parts], dim=0) for i in range(3)]

    return batch, mode


def evaluate(params, model, dataloader):
    model.eval()
    loss = 0
    loss_mode = defaultdict(float)
    mode_counts = Counter()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch, batch_mode = prepare_batch(params, model, batch)
            batch_loss = model(*batch).item()

            loss += batch_loss
            loss_mode[batch_mode] += batch_loss
            mode_counts[batch_mode] += 1

    loss /= len(dataloader)
    for mode in loss_mode.keys():
        loss_mode[mode] /= mode_counts[mode]
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
        mode_counts = Counter()

        for i, batch in tqdm(enumerate(dataloader_training)):
            batch, batch_mode = prepare_batch(params, model, batch)

            loss = model(*batch)
            loss = loss / params['gradient_accumulation']
            accelerator.backward(loss)

            step_loss += loss.item()
            step_loss_mode[batch_mode] += (loss.item() * params['gradient_accumulation'])
            mode_counts[batch_mode] += 1

            if (i + 1) % params['gradient_accumulation'] == 0:
                optimizer.step()
                optimizer.zero_grad()

                for mode in step_loss_mode.keys():
                    step_loss_mode[mode] /= mode_counts[mode]

                logger.info(f'step {global_step}, {mode_counts}: loss: {step_loss}, '
                            f'loss by mode: {dict(step_loss_mode)}')
                writer.add_scalar('loss/train', step_loss, global_step)

                for mode, loss in step_loss_mode.items():
                    writer.add_scalar(f'loss/train/{mode}', loss, global_step)

                step_loss = 0.0
                step_loss_mode = defaultdict(float)
                mode_counts = Counter()
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
