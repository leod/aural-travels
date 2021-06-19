import torch
from torch import nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from aural_travels.model.transformer_nat import TransformerNAT


class AudioDALLENAT(nn.Module):
    def __init__(self,
                 image_repr,
                 audio_seq_len,
                 audio_num_features,
                 control_num_features,
                 hidden_size, 
                 num_layers,
                 num_heads,
                 attention_dropout,
                 ffnn_dropout,
                 axial_attention):
        super().__init__()

        self.image_repr = image_repr
        self.audio_seq_len = audio_seq_len
        self.audio_num_features = audio_num_features
        self.control_num_features = control_num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffnn_dropout = ffnn_dropout

        for param in image_repr.parameters():
            param.requires_grad = False

        self.image_seq_len = self.image_repr.grid_size() ** 2

        self.image_emb = nn.Embedding(self.image_repr.vocab_size() + 1, hidden_size) # +1 for <bos>
        self.audio_pos_emb = nn.Embedding(audio_seq_len + 1, hidden_size) # +1 for <bos>

        self.image_pos_emb = AxialPositionalEmbedding(hidden_size,
                                                      axial_shape=(image_repr.grid_size(),
                                                                   image_repr.grid_size()))

        self.control_embs = nn.ModuleList(nn.Embedding(128, hidden_size) for _
                                          in range(control_num_features))
        self.transformer = TransformerNAT(hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          context_len=self.audio_seq_len + control_num_features + 1,
                                          grid_size=self.image_repr.grid_size(),
                                          axial_attention=axial_attention)
        self.input = nn.Linear(audio_num_features, hidden_size)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, image_repr.vocab_size()))

    def _audio_input(self, audio_seq):
        assert audio_seq.shape[1] == self.audio_seq_len
        assert audio_seq.shape[2] == self.audio_num_features

        audio_emb = self.input(audio_seq)
        audio_emb += self.audio_pos_emb(torch.arange(audio_emb.shape[1], device=audio_emb.device))

        return audio_emb

    def _image_input(self, image_seq):
        # We place a <bos> token between audio and image to kick off image token prediction.
        image_seq_with_bos = F.pad(image_seq, (1, 0), value=self.image_repr.vocab_size())
        image_emb = self.image_emb(image_seq_with_bos)
        image_emb[:, 1:, :] += self.image_pos_emb(image_emb[:, 1:, :])
        image_emb[:, 0, :] += self.audio_pos_emb(torch.tensor(self.audio_seq_len,
                                                              device=image_seq.device,
                                                              dtype=torch.long))
        return image_emb

    def _input(self, audio_seq, control, input_image_seq):
        audio_emb = self._audio_input(audio_seq)
        control_emb = torch.cat([self.control_embs[i](control[:, i] + 64)[:, None, :]
                                 for i in range(control.shape[1])],
                                dim=1)
        input_image_emb = self._image_input(input_image_seq)

        return torch.cat((audio_emb, control_emb, input_image_emb), dim=1)

    def forward(self, audio_seq, control, input_image_seq, target_image_seq, return_logits=False):
        output = self.transformer(self._input(audio_seq, control, input_image_seq))
        output = output[:, audio_emb.shape[1] + self.control_num_features + 1:, :]

        logits = self.output(output)
        logits_for_ce = torch.transpose(logits, 1, 2)

        loss = F.cross_entropy(logits_for_ce, target_image_seq)

        if return_logits:
            return loss, logits
        else:
            return loss

    @torch.no_grad()
    def generate_image_seq(self,
                           audio_seq,
                           control,
                           input_image_seq,
                           temperature=1.0,
                           top_k=0,
                           return_logits=False,
                           map_logits=None):
        audio_emb = self._audio_input(audio_seq) 

        output = self.transformer(self._input(audio_seq, control, input_image_seq))
        output = output[:, self.audio_seq_len + self.control_num_features + 1:, :]

        logits = self.output(output)
        if map_logits:
            logits = map_logits(logits)
        new_logits = torch.clone(logits)

        if top_k > 0:
            indices_to_remove = new_logits < torch.topk(new_logits, top_k)[0][..., -1, None]
            new_logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(new_logits / temperature, dim=-1)

        probs = rearrange(probs, 'b n v -> (b n) v')
        image_seq = torch.multinomial(probs, 1)[..., 0]
        image_seq = rearrange(image_seq, '(b n) -> b n', b=audio_seq.shape[0])

        if return_logits:
            return image_seq, logits
        else:
            return image_seq
