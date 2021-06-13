from abc import ABC, abstractmethod

import numpy as np
import PIL

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dalle_pytorch import OpenAIDiscreteVAE
from dalle_pytorch.vae import download

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel


class ImageRepr(nn.Module, ABC):
    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def grid_size(self):
        pass

    @abstractmethod
    def image_to_tensor(self, image):
        pass

    @abstractmethod
    def encode(self, image_tensor):
        pass

    @abstractmethod
    def decode(self, image_seq):
        pass

    @abstractmethod
    def tensor_to_image(self, image_tensor):
        pass

    def rand_image_seq(self, batch_size, patch_size=1, device=None):
        sample_size = self.grid_size() // patch_size
        sample = torch.randint(self.vocab_size(),
                               (batch_size, 1, sample_size, sample_size),
                               device=device)
        image_seq = F.upsample(sample.float(), scale_factor=(patch_size, patch_size))
        return image_seq.long().view(batch_size, -1)

    def zeros_image_seq(self, batch_size, device):
        return torch.zeros((batch_size, self.grid_size()**2), dtype=torch.long, device=device)


class DALLEImageRepr(ImageRepr):
    def __init__(self):
        super().__init__()

        self.vae = OpenAIDiscreteVAE()
        self.vae.eval()

    def vocab_size(self):
        return self.vae.num_tokens

    def grid_size(self):
        return self.vae.image_size // (2 ** self.vae.num_layers)

    def image_to_tensor(self, image):
        # As here:
        # https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
        s = min(image.size)
        
        if s < self.vae.image_size:
            raise ValueError(f'min dim for image {s} < {self.vae.image_size}')
            
        r = self.vae.image_size / s
        s = (round(r * image.size[1]), round(r * image.size[0]))
        image = TF.resize(image, s, interpolation=TF.InterpolationMode.LANCZOS)
        image = TF.center_crop(image, output_size=2 * [self.vae.image_size])
        tensor = transforms.ToTensor()(image)[None, ...]

        # Leaving out map_pixels here, since it is handled by dalle_pytorch.
        return tensor

    @torch.no_grad()
    def encode(self, image_tensor):
        return self.vae.get_codebook_indices(image_tensor)

    @torch.no_grad()
    def decode(self, image_seq):
        return self.vae.decode(image_seq)

    def tensor_to_image(self, image_tensor):
        # unmap_pixels is handled by dalle_pytorch.
        return T.ToPILImage(mode='RGB')(image_tensor)


class VQGANImageRepr(ImageRepr):
    # Codebook with 1024 entries
    CHECKPOINT_URL = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
    CHECKPOINT_FILENAME = 'vqgan_imagenet_f16_1024.ckpt'
    CONFIG_URL = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'
    CONFIG_FILENAME = 'vqgan_imagenet_f16_1024.yaml'

    def __init__(self):
        super().__init__()

        checkpoint_path = download(self.CHECKPOINT_URL, self.CHECKPOINT_FILENAME)
        config_path = download(self.CONFIG_URL, self.CONFIG_FILENAME)

        # Reference:
        # https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb

        config = OmegaConf.load(config_path)
        state_dict = torch.load(checkpoint_path)['state_dict']

        self.model = VQModel(**config.model.params)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def vocab_size(self):
        return 1024

    def grid_size(self):
        return 16

    def image_to_tensor(self, image):
        s = min(image.size)
        target_image_size = 256

        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')

        r = target_image_size / s
        s = (round(r * image.size[1]), round(r * image.size[0]))
        image = TF.resize(image, s, interpolation=TF.InterpolationMode.LANCZOS)
        image = TF.center_crop(image, output_size=2 * [target_image_size])
        tensor = T.ToTensor()(image)[None, ...]

        return tensor * 2 - 1

    @torch.no_grad()
    def encode(self, image_tensor):
        assert len(image_tensor.shape) == 4

        _, _, (_, _, indices) = self.model.encode(image_tensor)
        return indices.view(image_tensor.shape[0], -1)

    @torch.no_grad()
    def decode(self, image_seq, image_seq2=None, alpha=None, topk=False):
        # Not much of an idea what is happening here with the shapes to be honest.
        # https://github.com/CompVis/taming-transformers/blob/8549d3aaa09446bafc26efa032157c04833ca3ff/taming/models/cond_transformer.py#L157

        assert len(image_seq.shape) == 2

        # ??? what ???
        bhwc = (image_seq.shape[0], 16, 16, 256)

        if not topk:
            emb = self.model.quantize.get_codebook_entry(image_seq.reshape(-1), shape=bhwc)
            if image_seq2 is not None:
                emb2 = self.model.quantize.get_codebook_entry(image_seq2.reshape(-1), shape=bhwc)
                return self.model.decode((1.0 - alpha) * emb + alpha * emb2)
            else:
                return self.model.decode(emb)
        else:
            assert image_seq2 is not None

            W = self.model.quantize.embedding.weight

            emb = W[image_seq]
            emb2 = W[image_seq2]
            x = (1.0 - alpha) * emb + alpha * emb2

            W2 = torch.tile(W[None, :, :], (256, 1, 1))
            x2 = torch.tile(x.view(-1, 256)[:, None, :], (1024, 1))

            d = (W2 - x2).pow(2).sum(dim=-1)
            _, idx = torch.topk(-d, k=1, dim=1)
            idx = idx.view(x.shape[0], 256)

            emb = self.model.quantize.get_codebook_entry(idx.reshape(-1), shape=bhwc)
            return self.model.decode(emb)

    def tensor_to_image(self, image_tensor):
        x = image_tensor.detach().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
        x = x.permute(1,2,0).numpy()
        x = (255*x).astype(np.uint8)
        x = PIL.Image.fromarray(x)
        if not x.mode == "RGB":
            x = x.convert("RGB")

        return x
