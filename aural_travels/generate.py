import math
import random

import numpy as np

import torch

def keyframes(model,
              mel,
              last_image_seq,
              fps,
              top_k=0,
              temperature=1.0,
              sample_rate=22050,
              hop_length=1024,
              num_mel_samples=42,
              noise=None,
              device=None):
    mel_frame_duration = hop_length / sample_rate

    time = 0.0
    duration = mel.shape[0] * mel_frame_duration
    time_step = 1.0 / fps

    while time < duration:
        mel_sample_idx = int(time / mel_frame_duration)
        mel_slice = mel[None, mel_sample_idx:mel_sample_idx+num_mel_samples].to(device)

        image_seq = model.generate_image_seq(mel_slice,
                                             top_k=top_k,
                                             temperature=temperature,
                                             corrupt_image_seq=last_image_seq)
        yield image_seq.clone()

        last_image_seq = image_seq
        noise(time, time + time_step, last_image_seq)

        time += time_step


def interpolate(image_repr, keyframes, interframes):
    last_keyframe = next(keyframes)

    for keyframe in keyframes:
        for i in range(interframes):
            yield image_repr.decode(last_keyframe, keyframe, alpha=i/interframes)[0]

        last_keyframe = keyframe


def cross_noise(image_repr, image_seq):
    x = image_repr.grid_size() // 2
    y = int(round(image_repr.grid_size() * 1/1.618))

    image_seq = image_seq.view(1, image_repr.grid_size(), image_repr.grid_size())

    image_seq[:, y, :] = torch.randint(image_repr.vocab_size(), (image_repr.grid_size(),))
    image_seq[:, :, x] = torch.randint(image_repr.vocab_size(), (image_repr.grid_size(),))


def onset_env_noise(image_repr,
                    onset_env,
                    time,
                    next_time,
                    image_seq,
                    power=2.0,
                    scale=1.0,
                    sample_rate=22050,
                    hop_length=512):
    idx = math.floor(time * sample_rate / hop_length)
    next_idx = math.ceil(next_time * sample_rate / hop_length)

    strength = np.max(onset_env[idx:next_idx])
    k = min(image_repr.grid_size() ** 2, max(0, int(scale * (strength ** power))))

    print('env_noise', time, strength, k)

    idxs = random.sample(list(range(image_repr.grid_size() ** 2)), k=k)
    for idx in idxs:
        image_seq[0, idx] = random.randint(0, image_repr.vocab_size()-1)


def beat_cross_noise(image_repr,
                     beats,
                     time,
                     next_time,
                     image_seq):
    beat_idx = 0
    while beat_idx < len(beats) and beats[beat_idx] < time:
        beat_idx += 1

    if beat_idx < len(beats) and beats[beat_idx] < next_time:
        print('cross_noise', time)
        cross_noise(image_repr, image_seq)
