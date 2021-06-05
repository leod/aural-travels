import pytest

import torch

from dalle_pytorch import OpenAIDiscreteVAE

from aural_travels.model import AudioDALLENAT

def test_transformer_nat_batch():
    vae = OpenAIDiscreteVAE()
    vae.eval()

    model = AudioDALLENAT(vae=vae,
                          audio_seq_len=10,
                          audio_num_features=20,
                          hidden_size=128,
                          num_layers=3,
                          num_heads=8,
                          attention_dropout=0.1,
                          ffnn_dropout=0.1)
    model.eval()

    with torch.no_grad():
        for b in [1, 4, 5, 16]:
            xa = torch.randn((b, 10, 20))
            xb = torch.randint(8192, (b, 32**2))
            xc = torch.randint(8192, (b, 32**2))

            y1 = model(xa, xb, xc, return_logits=True)[1]
            y2 = model(xa, xb, xc, return_logits=True)[1]
            y3 = torch.cat(
                [model(rowa[None, ...], rowb[None, ...], rowc[None, ...], return_logits=True)[1]
                 for rowa, rowb, rowc in zip(xa, xb, xc)], dim=0)

            assert torch.allclose(y1, y2, atol=1e-6)
            assert torch.allclose(y1, y3, atol=1e-6)

            seq1 = model.generate_image_seq(xa, xb, top_k=1)
            seq2 = model.generate_image_seq(xa, xb, top_k=1)
            seq3 = torch.cat(
                [model.generate_image_seq(rowa[None, ...], rowb[None, ...], top_k=1)
                 for rowa, rowb in zip(xa, xb)], dim=0)

            assert torch.equal(seq1, seq2)
            assert torch.equal(seq1, seq3)
