import math
import random

import numpy as np

import torch


def keyframes(model,
              mel,
              last_image_seq,
              fps,
              top_k=0,
              temperature=lambda time, next_time: 1.0,
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

        tau = temperature(time, time + time_step)
        print('temp', tau)

        image_seq = model.generate_image_seq(mel_slice,
                                             top_k=top_k,
                                             temperature=tau,
                                             corrupt_image_seq=last_image_seq)
        yield image_seq.clone()

        last_image_seq = image_seq
        time += time_step

        noise(time, time + time_step, last_image_seq)


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

    #image_seq[:, y-1:y+1, :] \
    #    = torch.randint(image_repr.vocab_size(), (1, 2, image_repr.grid_size()))
    #image_seq[:, :,x-1:x+1] \
    #    = torch.randint(image_repr.vocab_size(), (1, image_repr.grid_size(), 2))

    image_seq[:, y, :] \
        = torch.randint(image_repr.vocab_size(), (image_repr.grid_size(),))
    image_seq[:, :,x] \
        = torch.randint(image_repr.vocab_size(), (image_repr.grid_size(),))


def onset_env_temperature(onset_env,
                          time,
                          next_time,
                          sample_rate=22050,
                          hop_length=512):
    idx = math.floor(time * sample_rate / hop_length)
    next_idx = math.ceil(next_time * sample_rate / hop_length)

    return min(1.6, max(1.0, np.mean(onset_env[idx:next_idx])))


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

    #strength = np.max(onset_env[idx:next_idx])
    strength = np.mean(onset_env[idx:next_idx])
    k = min(image_repr.grid_size() ** 2, max(0, int(scale * (strength ** power))))

    print('env_noise', time, strength, k)

    idxs = random.sample(list(range(image_repr.grid_size() ** 2)), k=k)
    for idx in idxs:
        image_seq[0, idx] = random.randint(0, image_repr.vocab_size()-1)


def onset_env_temperature(onset_env,
                          time,
                          next_time,
                          sample_rate=22050,
                          hop_length=512):
    idx = math.floor(time * sample_rate / hop_length)
    next_idx = math.ceil(next_time * sample_rate / hop_length)

    return min(1.6, max(1.0, np.mean(onset_env[idx:next_idx])))


def onset_env_bump_noise(image_repr,
                         onset_env,
                         time,
                         next_time,
                         image_seq,
                         sample_rate=22050,
                         hop_length=512):
    idx = math.floor(time * sample_rate / hop_length)
    next_idx = math.ceil(next_time * sample_rate / hop_length)

    strength = np.mean(onset_env[idx:next_idx])
    size = max(0, int(round((strength - 1) * 8)))

    print('bump_noise', time, strength, size)
    image_seq = image_seq.view(1, image_repr.grid_size(), image_repr.grid_size())

    for dx in range(size):
        for dy in range(size):
            x = int(round((image_repr.grid_size() - size) / 2.0)) + dx
            y = int(round((image_repr.grid_size() - size) / 2.0)) + dy

            image_seq[0, x, y] = random.randint(0, image_repr.vocab_size() - 1)


def onset_env_circle_noise(image_repr,
                           onset_env,
                           state,
                           time,
                           next_time,
                           image_seq,
                           sample_rate=22050,
                           hop_length=512):
    idx = math.floor(time * sample_rate / hop_length)
    next_idx = math.ceil(next_time * sample_rate / hop_length)

    strength = np.mean(onset_env[idx:next_idx])

    if state is None:
        state = 0.0

    delta = 8.0 * max(0.0, strength - 0.5) / math.pi
    
    print('circle_noise', state, strength, delta)

    image_seq = image_seq.view(1, image_repr.grid_size(), image_repr.grid_size())

    color = random.randint(0, image_repr.vocab_size() - 1)

    while delta > 0.0:
        state += delta
        delta -= 0.1

        dx = math.floor(7.5 * math.cos(-state))
        dy = math.floor(7.5 * math.sin(-state))

        x = 8 + dx
        y = 8 + dy

        image_seq[0, x, y] = color #random.randint(0, image_repr.vocab_size() - 1)

    return state
    

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


def segment_reset_noise(image_repr,
                        boundaries,
                        time,
                        next_time,
                        image_seq):
    segment_idx = 0
    while segment_idx < len(boundaries) and boundaries[segment_idx] < time:
        segment_idx += 1

    if segment_idx < len(boundaries) and boundaries[segment_idx] < next_time:
        print('reset_noise', time)
        image_seq[:, :] = image_repr.rand_image_seq(1, patch_size=2)

