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

from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from accelerate import Accelerator

from aural_travels.model import audio_dalle, AudioDALLE, AudioDALLENAT
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
    if params['non_autoregressive']:
        model = AudioDALLENAT(image_repr=image_repr,
                              audio_seq_len=dataset.num_samples(),
                              audio_num_features=dataset.num_features(),
                              control_num_features=3,
                              hidden_size=params['hidden_size'],
                              num_layers=params['num_layers'],
                              num_heads=params['num_heads'],
                              attention_dropout=params['attention_dropout'],
                              ffnn_dropout=params['ffnn_dropout'],
                              axial_attention=params['axial_attention'])
    else:
        # TODO: Untested after image_repr refactoring
        model = AudioDALLE(image_repr=image_repr,
                           audio_seq_len=dataset.num_samples(),
                           audio_num_features=dataset.num_features(),
                           hidden_size=params['hidden_size'],
                           num_layers=params['num_layers'],
                           num_heads=params['num_heads'],
                           attention_dropout=params['attention_dropout'],
                           ffnn_dropout=params['ffnn_dropout'])

    return model


def apply_corruption(mode, vocab_size, image_seq):
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


def resize(image):
    s = min(image.size)
    r = 256 / s
    s = round(r * image.size[1]), round(r * image.size[0])
    #image = TF.resize(image, s, interpolation=TF.InterpolationMode.LANCZOS)
    image = TF.resize(image, s, interpolation=TF.InterpolationMode.NEAREST)

    return image


def zoom_pair(image_repr, item):
    image = item['image']

    width, height = image.size[0], image.size[1]
    assert width == height
    
    size1 = random.randint(1, width)
    
    x1 = random.uniform(size1/2, width - size1/2)
    y1 = random.uniform(size1/2, height - size1/2)

    if random.random() < 0.2:
        margin1 = 0
    else:
        margin1 = min(x1 - size1/2,
                      y1 - size1/2,
                      width - (x1 + size1/2),
                      height - (y1 + size1/2),
                      1.5 * size1,
                      64)
        
    expand = random.uniform(0, margin1)
    size2 = size1 + expand
    
    if random.random() < 0.2:
        dx = 0
        dy = 0
    else:
        min_dx = -min(x1 - size2/2, 0.5 * size1, 64)
        max_dx = min(width - (x1 + size2/2), 0.5 * size1, 64)
        
        min_dy = -min(y1 - size2/2, 0.5 * size1, 64)
        max_dy = min(width - (y1 + size2/2), 0.5 * size1, 64)
        
        dx = random.uniform(min_dx, max_dx)
        dy = random.uniform(min_dy, max_dy)
    
    x2 = x1 + dx
    y2 = y1 + dy
    
    image1 = TF.crop(image, top=y1 - size1/2, left=x1 - size1/2, height=size1, width=size1)
    image2 = TF.crop(image, top=y2 - size2/2, left=x2 - size2/2, height=size2, width=size2)
    
    #print(f'expand={expand:.4f}, dx={dx:.4f}, dy={dy:.4f}, margin1={margin1:.4f}, size1={size1}, size2={size2}')

    del item['image']
    item['input_image'] = image_repr.image_to_tensor(resize(image1))[0]
    item['target_image'] = image_repr.image_to_tensor(resize(image2))[0]
    item['control'] = torch.tensor([dx, dy, expand], dtype=torch.long)


def prepare_batch(params, model, batch):
    batch['input_image_seq'] = model.image_repr.encode(batch['input_image']) 
    batch['target_image_seq'] = model.image_repr.encode(batch['target_image']) 
    del batch['input_image']
    del batch['target_image']

    return batch, 'zoom_pair'


def load_dataset(params, image_repr, split):
    def map_item(item):
        zoom_pair(image_repr, item)
        return item 

    return soundcloud.CoverGenerationDataset(data_dir=params['soundcloud_data_dir'],
                                             split=split,
                                             sample_secs=params['sample_secs'],
                                             n_fft=params['n_fft'],
                                             hop_length=params['hop_length'],
                                             toy_data=params['toy_data'],
                                             map_item=map_item)


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


def evaluate(params, model, dataloader):
    model.eval()
    loss = 0
    loss_mode = defaultdict(float)
    mode_counts = Counter()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch, batch_mode = prepare_batch(params, model, batch)
            batch_loss = model(**batch).item()

            loss += batch_loss
            loss_mode[batch_mode] += batch_loss
            mode_counts[batch_mode] += 1

    loss /= len(dataloader)
    for mode in loss_mode.keys():
        loss_mode[mode] /= mode_counts[mode]
    return loss, loss_mode


def train(params, model, optimizer, dataloaders):
    # Need to reimplement this.
    assert params['expose_steps'] is None
    assert params['corrupt_image_mode'] is None

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

            loss = model(**batch)
            loss = loss / params['gradient_accumulation']
            accelerator.backward(loss)

            step_loss += loss.item()
            step_loss_mode[batch_mode] += loss.item() * params['gradient_accumulation']
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
