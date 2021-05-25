import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision.transforms.functional as TF

from axial_positional_embedding import AxialPositionalEmbedding
from torch.nn.modules import sparse
from dalle_pytorch.transformer import Transformer


def transform_image(image_size, image):
    # As here:
    # https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
    s = min(image.size)
    
    if s < image_size:
        raise ValueError(f'min dim for image {s} < {image_size}')
        
    r = image_size / s
    s = (round(r * image.size[1]), round(r * image.size[0]))
    image = TF.resize(image, s, interpolation=TF.InterpolationMode.LANCZOS)
    image = TF.center_crop(image, output_size=2 * [image_size])
    image = transforms.ToTensor()(image)

    # NOTE: Leaving out map_pixels here, since it is handled by dalle_pytorch.
    return image


class AudioDALLE(nn.Module):
    # Closely following this:
    # <https://github.com/lucidrains/DALLE-pytorch/blob/bdb04280c9ab55eb20f86b375dc1aad20fbd5315/dalle_pytorch/dalle_pytorch.py#L308>,
    # but with continuous audio features rather than discrete tokens as input.

    def __init__(self,
                 vae,
                 audio_seq_len,
                 audio_num_features,
                 hidden_size, 
                 num_layers,
                 num_heads,
                 attention_dropout,
                 ffnn_dropout):
        super().__init__()

        self.vae = vae
        self.audio_seq_len = audio_seq_len
        self.audio_num_features = audio_num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffnn_dropout = ffnn_dropout

        for param in vae.parameters():
            param.requires_grad = False

        self.grid_size = vae.image_size // (2 ** vae.num_layers)
        self.image_seq_len = self.grid_size ** 2
        self.total_seq_len = audio_seq_len + self.image_seq_len

        # Sparse attention order as in the DALL-E paper.
        attention_types = ['axial_col' if (i - 2) % 4 == 0 else 'axial_row'
                           for i in range(num_layers - 1)]
        attention_types += ['conv_like']

        self.image_emb = nn.Embedding(vae.num_tokens + 1, hidden_size) # +1 for <bos>
        self.audio_pos_emb = nn.Embedding(audio_seq_len + 1, hidden_size) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(hidden_size,
                                                      axial_shape=(self.grid_size, self.grid_size))

        self.transformer = Transformer(dim=hidden_size,
                                       causal=True,
                                       seq_len=self.total_seq_len,
                                       depth=num_layers,
                                       heads=num_heads,
                                       dim_head=hidden_size // num_heads,
                                       reversible=False,
                                       attn_dropout=attention_dropout,
                                       ff_dropout=ffnn_dropout,
                                       attn_types=attention_types,
                                       image_fmap_size=self.grid_size,
                                       sparse_attn=True)
        self.input = nn.Linear(audio_num_features, hidden_size)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, vae.num_tokens))

    def _audio_input(self, audio):
        assert audio.shape[1] == self.audio_seq_len
        assert audio.shape[2] == self.audio_num_features

        audio_emb = self.input(audio)
        audio_emb += self.audio_pos_emb(torch.arange(audio_emb.shape[1], device=audio_emb.device))

        return audio_emb

    def _image_input(self, image):
        # We place a <bos> token between audio and image to kick off image token prediction.
        image_with_bos = F.pad(image, (1, 0), value=self.vae.num_tokens)
        image_emb = self.image_emb(image_with_bos)
        image_emb[:, 1:, :] += self.image_pos_emb(image_emb[:, 1:, :])
        image_emb[:, 0, :] += self.audio_pos_emb(torch.tensor(self.audio_seq_len,
                                                              device=image.device,
                                                              dtype=torch.long))
        return image_emb

    def forward(self, audio, image):
        audio_emb = self._audio_input(audio)
        image_emb = self._image_input(image)

        input = torch.cat((audio_emb, image_emb[:, :-1]), dim=1)

        output = self.transformer(input)
        output = output[:, audio_emb.shape[1]:, :]

        logits = self.output(output)
        logits = torch.transpose(logits, 1, 2)

        loss = F.cross_entropy(logits, image)

        return loss

    @torch.no_grad()
    def generate_images(self,
                        audio,
                        temperature=1.0,
                        top_k=0,
                        map_logits=None):
        audio_emb = self._audio_input(audio) 

        image_seq = torch.zeros((audio.size()[0], 0), dtype=torch.long, device=audio.device)

        for step in range(self.image_seq_len):
            image_emb = self._image_input(image_seq)
            input = torch.cat((audio_emb, image_emb_seq), dim=1)

            output = self.transformer(input)
            logits = self.output(output)

            if map_logits:
                logits = map_logits(logits)

            new_logits = torch.clone(logits[:, -1, :])

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            image_seq = torch.cat((image_seq, sample), dim=-1)

        return image, logits #self.vae.decode(image)
