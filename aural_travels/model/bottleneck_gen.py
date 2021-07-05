import logging

import torch
from torch import nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from aural_travels.model.transformer_nat import TransformerNAT


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BottleneckGen(nn.Module):
    def __init__(self,
                 image_repr,
                 audio_seq_len,
                 audio_num_features,
                 hidden_size,
                 num_enc_layers,
                 num_dec_layers,
                 num_heads,
                 attention_dropout,
                 ffnn_dropout,
                 input_dropout,
                 audio_emb_dropout,
                 use_layer_scale,
                 num_latents,
                 latent_size):
        super().__init__()

        self.image_repr = image_repr
        self.audio_seq_len = audio_seq_len
        self.audio_num_features = audio_num_features
        self.hidden_size = hidden_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffnn_dropout = ffnn_dropout
        self.input_dropout = nn.Dropout(input_dropout)
        self.audio_emb_dropout = nn.Dropout(audio_emb_dropout)
        self.num_latents = num_latents
        self.latent_size = latent_size

        for param in image_repr.parameters():
            param.requires_grad = False

        self.image_seq_len = self.image_repr.grid_size() ** 2

        self.audio_input = nn.Linear(audio_num_features, hidden_size)
        self.audio_pos_emb = nn.Embedding(audio_seq_len, hidden_size)
        self.audio_encoder = TransformerNAT(hidden_size=hidden_size,
                                            num_layers=num_enc_layers,
                                            use_layer_scale=use_layer_scale)

        self.image_pos_emb = AxialPositionalEmbedding(hidden_size,
                                                      axial_shape=(image_repr.grid_size(),
                                                                   image_repr.grid_size()))
        self.image_decoder = TransformerNAT(hidden_size=hidden_size,
                                            num_layers=num_dec_layers) 
        self.image_output = nn.Sequential(nn.LayerNorm(hidden_size),
                                          nn.Linear(hidden_size, image_repr.vocab_size()))

        if num_latents > 0:
            self.latents = nn.Embedding(num_latents, latent_size)
            self.latents.weight.requires_grad = False
            self.latent_input_encoder = nn.Linear(latent_size, hidden_size)
            self.latent_input_decoder = nn.Linear(latent_size, hidden_size)

    def _audio_input(self, audio_seq):
        assert audio_seq.shape[1] == self.audio_seq_len
        assert audio_seq.shape[2] == self.audio_num_features

        audio_seq = self.input_dropout(audio_seq)
        audio_emb = self.audio_input(audio_seq)
        audio_emb += self.audio_pos_emb(torch.arange(audio_emb.shape[1], device=audio_emb.device))

        return audio_emb

    def forward(self, audio_seq, target_image_seq, audio_seq2=None):
        if self.num_latents > 0:
            batch_size = audio_seq.shape[0]

            audio_input = self._audio_input(audio_seq)
            audio_input = torch.repeat_interleave(audio_input, self.num_latents, dim=0)
            encoder_latents = self.latent_input_encoder(self.latents.weight)
            encoder_latents = torch.tile(encoder_latents, (batch_size, 1))[:, None, :]

            audio_emb = self.audio_encoder(audio_input + encoder_latents)
            audio_emb = torch.mean(audio_emb, dim=1)
            audio_emb = self.audio_emb_dropout(audio_emb)
            decoder_latents = self.latent_input_decoder(self.latents.weight)
            decoder_latents = torch.tile(decoder_latents, (batch_size, 1))

            logits = self.calc_logits(audio_emb + decoder_latents)
            logits = torch.transpose(logits, 1, 2)

            target_image_seq = torch.repeat_interleave(target_image_seq, self.num_latents, dim=0)
            generate_loss = F.cross_entropy(logits, target_image_seq, reduction='none')

            generate_loss = generate_loss.mean(dim=-1)
            generate_loss = generate_loss.view(batch_size, self.num_latents)
            generate_loss_amin = torch.amin(generate_loss, dim=1).mean().item()
            generate_loss = -torch.logsumexp(-generate_loss, dim=1).mean()
        else:
            audio_emb = self.calc_audio_emb(audio_seq)

            logits = self.calc_logits(audio_emb)
            logits = torch.transpose(logits, 1, 2)
            generate_loss = F.cross_entropy(logits, target_image_seq)

        if audio_seq2 is None:
            return generate_loss, generate_loss_amin
        else:
            assert False
        #    audio_emb2 = self.calc_audio_emb(audio_seq2)
        #    logits2 = self.calc_logits(audio_emb2)
        #    logits2 = torch.transpose(logits2, 1, 2)
        #    generate_loss2 = F.cross_entropy(logits2, target_image_seq)

        #    generate_loss = (generate_loss1 + generate_loss2) / 2.0

        #    pull_loss = F.mse_loss(audio_emb1, audio_emb2)

        #    audio_emb1 = 5.0 * F.normalize(audio_emb1)
        #    audio_emb2 = 5.0 * F.normalize(audio_emb2)
        #    cosine_sims = torch.matmul(audio_emb1, audio_emb2.t())

        #    targets = torch.arange(audio_seq1.shape[0], device=audio_seq1.device)
        #    contrastive_loss1 = F.cross_entropy(cosine_sims, targets)
        #    contrastive_loss2 = F.cross_entropy(cosine_sims.t(), targets)
        #    contrastive_loss = (contrastive_loss1 + contrastive_loss2) / 2.0

        #    return generate_loss, contrastive_loss, pull_loss

    def calc_logits(self, audio_emb):
        audio_emb = torch.tile(audio_emb[:, None, :], (self.image_seq_len, 1))
        audio_emb += self.image_pos_emb(audio_emb)
        image_emb = self.image_decoder(audio_emb)
        return self.image_output(image_emb)

    def calc_audio_emb(self, audio_seq, latent=None):
        audio_input = self._audio_input(audio_seq)
        if latent is not None:
            audio_input = audio_input + self.latent_input_encoder(self.latents(latent))

        audio_emb = self.audio_encoder(audio_input)
        audio_emb = torch.mean(audio_emb, dim=1)
        return self.audio_emb_dropout(audio_emb)

    def generate_image_seq(self,
                           audio_emb,
                           temperature=1.0,
                           top_k=1,
                           latent=None):
        if latent is not None:
            audio_emb = audio_emb + self.latent_input_decoder(self.latents(latent))

        logits = self.calc_logits(audio_emb)

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits / temperature, dim=-1)

        probs = rearrange(probs, 'b n v -> (b n) v')
        image_seq = torch.multinomial(probs, 1)[..., 0]
        image_seq = rearrange(image_seq, '(b n) -> b n', b=logits.shape[0])

        return image_seq
